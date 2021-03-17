import cv2
import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator as RGI
from dataclasses import dataclass, fields


@dataclass
class ImageVolume:
    # A single volume, single modality reconstruction
    # This would be returned in the following:
    # series = Series(files)
    # ct_vol = series.reconstruct.ct()
    # struct = series.reconstruct.struct()
    array: np.array
    dicom_header: pydicom.dataset.Dataset
    dimension: list = None
    pixelspacing: list = None
    slicethickness: float = None
    interpolated: bool = False
    missing_slices: list = None

    def __post_int__(self):
        self.pixelspacing = self.dicom_header.PixelSpacing
        self.slicethickness = self.dicom_header.SliceThickness
        self.dicom_header = None


@dataclass
class StructVolumeSet:
    volumes: dict  # Named as the structures and the volumes
    dimension: list = None
    pixelspacing: list = None
    slicethickness: float = None
    interpoltaed: bool = False
    missing_slices: list = None

    def __post_int__(self):
        self.pixelspacing = self.dicom_header.PixelSpacing
        self.slicethickness = self.dicom_header.SliceThickness
        self.dicom_header = None


@dataclass
class VolumeDimensions:
    dicoms: list
    origin: list = None
    rows: int = None
    cols: int = None
    slices: int = None
    dims: list = None
    dx: float = None
    dy: float = None
    dz: float = None
    flipped: bool = False

    def __post_init__(self):
        if self.dicoms[0].Modality == 'RTDOSE':
            ds = pydicom.dcmread(self.dicoms[0].filepath)
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePatientPosition
        else:
            for dcm in self.dicoms:
                ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
                ipp = ds.ImagePatientPosition
                if ds.InstanceNumber == 1:
                    self._z0 = float(ipp[-1])
                if ds.InstanceNumber == self.dicoms.nfiles:
                    self._z1 = float(ipp[-1])

            self.origin = np.array([*ipp[:2], min(self._z0, self._z1)])
            self._tot_z = max(self._z0, self._z1) - min(self._z0, self._z1)
            self.slices = self._tot_z / self.dz
            if self._z1 > self._z0:
                self.flipped = True

        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = ds.PixelSpacing
        self.dz = ds.SliceThickness

    @property
    def shape(self):
        return (self.rows, self.cols, self.slices)

    def coordrange(self):
        pts_x = self.origin[0] + np.arange(self.rows) * self.dx
        pts_y = self.origin[1] + np.arange(self.cols) * self.dy
        pts_z = self.origin[2] + np.arange(self.slices) * self.dz
        if self.flipped:
            pts_z = pts_z[..., ::-1]
        return (pts_x, pts_y, pts_z)

    def coordgrid(self):
        pts_x, pts_y, pts_z = self.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)


class Reconstruction:
    # Reconstruction only works at the Series level
    # We can merge series to create combo ones
    # Will iteratively reconstruct each Modality group
    # Return a ImageVolume dataclass
    #def __init__(self, series):
        # We want this to inherit from the Series
        #self.dims = None # VolumeDimensions(series)

    def _point_to_coords(self, contour_pts, ct_group):
        img_origin = ct_group.origin
        vox_size = [*ct_group.PixelSpacing, abs(ct_group.SliceThickness)]
        diff = abs(contour_pts - img_origin)
        points = np.array(np.round(diff / vox_size), dtype=np.int32)
        return points

    def _build_contour(self, contour_data, ct_group):
        fill_array = np.zeros(ct_group.dims)
        for contour_slice in contour_data:
            contour_pts = np.array(contour_slice.ContourData).reshape(-1, 3)

            points = self._points_to_coords(contour_pts, ct_group)
            coords = np.array([points[:, :2]], dtype=np.int32)

            poly2D = np.zeros(ct_group.dims[:2])
            cv2.fillPoly(poly2D, coords, 1)

            fill_array[:, :, points[0, 2]] += poly2D
        fill_array = fill_array % 2
        return fill_array

    def struct(self, series):
        # Return as a struct volume type
        struct_sets = []
        for struct_group in series.struct:
            struct_set = StructVolumeSet()
            ref_group = series.frame_of_ref_subset(struct_group.FrameOfRef)
            ct_group = ref_group.ct[0]
            for struct_file in struct_group:
                ds = pydicom.dcmread(struct_file)
                for index, contour in enumerate(ds.StructureSetROISequence):
                    name = contour.ROIName.lower()
                    contour_data = ds.ROIContourSequence[index].ContourSequence
                    built = self._build_contour(contour_data, ct_group)
                    struct_set.add(built, name)
            struct_sets.append(struct_set)
        return struct_sets

    def ct(self, series, HU=True):
        ct_vols = []
        for ct_group in series.ct:
            fill_array = np.zeros(ct_group.dims, dtype='float32')
            origin = ct_group.origin

            for ct_file in ct_group:
                ds = pydicom.dcmread(ct_file)
                z_loc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, z_loc] = ds.pixel_array

            if HU:
                ct_vols.append(fill_array * ds.RescaleSlope +
                               ds.RescaleIntercept)
            ct_vols.append(fill_array)
        return ct_vols

    def nm(self, series):
        nm_vols = []
        for nm_group in series.nm:
            ds = pydicom.dcmread(nm_group[0])
            raw = np.rollaxis(ds.pixel_array, 0, 3)[:, :, ::-1]
            map_seq = ds.RealWorldValueMappingSequence[0]
            slope = map_seq.RealWorldValueSlope
            intercept = map_seq.RealWorldValueIntercept
            nm_vols.append(raw * slope + intercept)
        return nm_vols

    def mr(self, series):
        mr_vols = []
        for mr_group in series.mr:
            fill_array = np.zeros(mr_group.dims)
            origin = mr_group.origin

            for mr_file in mr_group:
                ds = pydicom.dcmread(mr_file)
                z_loc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, z_loc] = ds.pixel_array
            mr_vols.append(fill_array)
        return mr_vols

    def pet(self, series):
        pet_vols = []
        for pet_group in series.pet:
            fill_array = np.zeros(pet_group.dims)
            origin = pet_group.origin

            for pet_file in pet_group:
                ds = pydicom.dcmread(pet_file)
                z_loc = int(round(abs((origin[-1]))))
                fill_array[:, :, z_loc] = ds.pixel_array

            rescaled = fill_array * ds.RescaleSlope + ds.RescaleIntercept
            patient_weight = 1000 * float(ds.PatientWeight)
            radiopharm = ds.RadiopharmaceuticalInformationSequence[0]
            total_dose = float(radiopharm.RadionuclideTotalDose)
            pet_vols.append(rescaled * patient_weight / total_dose)
        return pet_vols

    def dose(self, series):
        dose_vols = {}
        ct_dims = VolumeDimensions(series.ct)
        ct_coords = ct_dims.coordgrid()
        for dosefile in series.dose:
            ds = pydicom.dcmread(dosefile.filepath)
            dose_dims = VolumeDimensions(dosefile)
            dose_array = np.rollaxis(ds.pixel_array, 0, 3) * ds.DoseGridScaling
            dose_coords = dose_dims.coordrange()
            interper = RGI(dose_coords, dose_array,
                           bounds_error=False, fill_value=0)
            dose_interp = interper(ct_coords).reshape(ct_dims.shape)
            dose_vols.update({dosefile: dose_interp})
        return dose_vols


class Deconstruction:
    def __init__(self):
        self.temp = None


class Tools:
    # Tools are used to process reconstructed arrays
    # DicomUtils are for DICOM reconstruction utilities
    def dose_max_points(self, dose_array, dose_coords=None):
        index = np.unravel_index(np.argmax(dose_array), dose_array.shape)
        if dose_coords:
            return dose_coords[index]
        return index


# Can reconstruct either via Series.recon.ct()
# or processing.recon.ct()
# This will require us to design the file structure accordingly
