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
class SeriesVolumeSet:
    # All reconstuctions for a given series
    # This is what will be returned by the call
    # series = Series(files)
    # volumes = series.reconstruct()
    modalities: dict
    series_uid: str
    study_uid: str
    patient_id: str


class Reconstruction:
    # Reconstruction only works at the Series level
    # We can merge series to create combo ones
    # Will iteratively reconstruct each Modality group
    # Return a ImageVolume dataclass
    def __init__(self):
        # We want this to inherit from the Series
        self.temp = None
        self.dims = None #self._vol_dims()

    def _vol_dims(self, frame_group):
        # For each should have this be at the group level???????????
        ct_files = frame_group.ct[0]
        z_min, z_max = ct_files.SliceRange
        n_slices = round(1 + (z_max - z_min) / ct_files.SliceThickness)
        return [n_slices, ct_files.cols, ct_files.rows]

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
        dose_arrays = []
        for dose_file in series.dose:
            ct = series.get_associated('ct', dose_file)
            dose_dcm = pydicom.dcmread(dose_file)
            ct_dcm = pydicom.dcmread(ct[0])

            img_origin = np.array(
                [*ct_dcm.ImagePositionPatient[:2], self.SliceRange[0]])

            img_dims = self.img_dims()
            ix, iy, iz = (*ct.PixelSpacing, ct.SliceThickness)

            dose_iso = np.array(dose_dcm.ImagePositionPatient)
            dose_dims = np.rollaxis(dose_dcm.pixel_array, 0, 3).shape
            dx, dy, dz = (*dose_dcm.PixelSpacing, dose_dcm.SliceThickness)

            d_grid_x = dose_iso[1] + dx * np.arange(dose_dims[0])
            d_grid_y = dose_iso[0] + dy * np.arange(dose_dims[1])
            d_grid_z = dose_iso[2] + dz * np.arange(dose_dims[2])

            i_grid_x = img_origin[1] + ix * np.arange(img_dims[0])
            i_grid_y = img_origin[0] + iy * np.arange(img_dims[1])
            i_grid_z = img_origin[2] + iz * np.arange(img_dims[2])

            grids = [(i_grid_x, d_grid_x), (i_grid_y,
                                            d_grid_y), (i_grid_z, d_grid_z)]
            list_of_grids = []

            def nearest(array, value):
                return np.abs(np.array(array) - value).argmin()

            # Compute the nearest CT coordinate to each dose coordinate
            for _, (img_grid, dose_grid) in enumerate(grids):
                temp = list(set([nearest(img_grid, val) for val in dose_grid]))
                temp.sort()
                list_of_grids.append(np.array(temp))

            dose_array = np.rollaxis(dose_dcm.pixel_array, 0, 3)

            x_mm, y_mm, z_mm = [[t[0], t[-1]] for t in list_of_grids]
            total_pts = np.product([[y - x] for x, y in [x_mm, y_mm, z_mm]])
            grid = np.mgrid[x_mm[0]: x_mm[1],
                            y_mm[0]: y_mm[1],
                            z_mm[0]: z_mm[1]].reshape(3, total_pts)
            interp_pts = np.squeeze(np.array([grid]).T)

            interp = RGI(list_of_grids, dose_array, method='linear')
            interp_vals = interp(interp_pts)
            interp_vol = interp_vals.reshape(x_mm[1] - x_mm[0],
                                             y_mm[1] - y_mm[0],
                                             z_mm[1] - z_mm[0])
            full_vol = np.zeros(img_dims)

            full_vol[x_mm[0]: x_mm[1],
                     y_mm[0]: y_mm[1],
                     z_mm[0]: z_mm[1]] = interp_vol

            dose_arrays.append(full_vol * float(dose_dcm.DoseGridScaling))
        return dose_arrays


class Deconstruction:
    def __init__(self):
        self.temp = None


class Tools:
    # Tools are used to process reconstructed arrays
    # DicomUtils are for DICOM reconstruction utilities
    def __init__(self):
        self.temp = None


# Can reconstruct either via Series.recon.ct()
# or processing.recon.ct()
# This will require us to design the file structure accordingly