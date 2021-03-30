import cv2
import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator as RGI
from dataclasses import dataclass, field, fields


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
    vox_size: list = None
    flipped: bool = False
    multi_thick: bool = False

    def __post_init__(self):
        if 'RTDOSE' in self.dicoms:
            filepath = self.dicoms['RTDOSE'][0].filepath
            ds = pydicom.dcmread(filepath)
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePositionPatient
            self.dz = float(ds.SliceThickness)
        else:
            if 'CT' in self.dicoms:
                files = self.dicoms['CT']
            elif 'MR' in self.dicoms:
                files = self.dicoms['MR']

            ds = self._calc_n_slices(files)

        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = map(float, ds.PixelSpacing)
        self.dims = [self.rows, self.cols, self.slices]
        self.position = ds.PatientPosition
        self.vox_size = [self.dx, self.dy, self.dz]
        self.dicoms = None

    def _calc_n_slices(self, files: list):
        """[calculates the number of volume slices]

        Args:
            files ([DicomFile]): [A list of DicomFile objects]

        Notes:
            Creating the volume by the difference in slice location at high and
            low instances ensures proper registration to rstructs, even if
            images slices are missing. We can interpolate to the lowest
            instance if we do not have instance 1, but we cannot extrapolate
            if higher instances are missing
        """
        inst0 = np.inf
        inst1 = -np.inf
        slice_thicknesses = []

        for dcm in files:
            ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
            ipp = ds.ImagePositionPatient
            inst = int(ds.InstanceNumber)
            slice_thicknesses.append(float(ds.SliceThickness))
            if inst < inst0:  # Low instance
                inst0 = inst
                z0 = float(ipp[-1])
            if inst > inst1:  # High instance
                inst1 = inst
                z1 = float(ipp[-1])

        if inst0 > 1:
            z0 -= ds.SliceThickness * (inst0 - 1)
            inst0 = 1

        slice_thicknesses = list(set(slice_thicknesses))
        if len(slice_thicknesses) > 1:
            self.multi_thick = True

        self.dz = min(slice_thicknesses)
        self.origin = np.array([*ipp[:2], min(z0, z1)])
        self.slices = round((max(z0, z1) - min(z0, z1)) / self.dz)

        if z1 > z0:
            # TODO: We can replace thiw with the ImagePositionPatient header
            self.flipped = True

        return ds

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


@dataclass
class StructVolumeSet:
    dims: VolumeDimensions
    volumes: dict = field(default_factory=dict)  # Named as the structures and the volumes
    interpoltaed: bool = False
    missing_slices: list = None

    def __getitem__(self, name: str):
        return self.volumes[name]

    def __setitem__(self, name: str, volume: np.array):
        if name in self.volumes:
            name = self._rename(name)
        self.volumes.update({name: volume})

    def _rename(self, name: str):
        """[DICOM RTSTRUCTs can have non-unique names, so we need to rename
            these functions to be dictionary compatiable]

        Args:
            name (str): [Name of the found RTSTRUCT]

        Returns:
            [str]: [Name + # where number is the unique occurance]
        """
        i = 0
        while True:
            temp = name + i
            if temp not in self.volumes:
                break
            i += 1
        return temp

    @property
    def shape(self):
        return (len(self.volumes), *self.dims.shape)

    @property
    def names(self):
        self.volumes = dict(sorted(self.volumes.items()))
        return list(self.volumes.keys())


@dataclass
class ImageVolume:
    # A single volume, single modality reconstruction
    # This would be returned in the following:
    # series = Series(files)
    # ct_vol = series.reconstruct.ct()
    # struct = series.reconstruct.struct()
    dims: VolumeDimensions
    array: np.array
    dicom_header: pydicom.dataset.Dataset
    interpolated: bool = False
    missing_slices: list = None


class Reconstruction:
    def __call__(self, frame_of_ref):
        self.define_vol_dims(frame_of_ref)
        for mod in frame_of_ref.iter_modalities:
            # TODO: Make switch statement in python 3.10
            """
            if mod.dirname == 'CT':
                vol = self.ct(mod)
            elif mod.dirname == 'MR':
                vol = self.mr(mod)
            elif mod.dirname == 'PET':
                vol = self.pet(mod)
            elif mod.dirname == 'DOSE':
                vol = self.dose(mod)
            """
            if mod.dirname == 'RTSTRUCT':
                vols = self.struct(mod)
                for vol in vols:
                    print(vol.shape, vol.names)

    def _slice_coords(self, contour_slice: pydicom.dataset.Dataset) -> (np.array, int):
        """[converts a dicom contour slice into image coordinates]

        Args:
            contour_slice (pydicom.dataset.Dataset): [(3006, 0016): Contour Image Sequence]

        Returns:
            [np.array]: [2D numpy array of contour points in image coordinates]
            [int]: [The z location of the rtstruct slice location]
        """
        contour_pts = np.array(contour_slice.ContourData).reshape(-1, 3)
        pts_diff = abs(contour_pts - self.dims.origin)
        points = np.array(np.round(pts_diff / self.dims.vox_size), dtype=np.int32)
        coords = np.array([points[:, :2]], dtype=np.int32)
        zloc = list(set(points[:, 2]))
        if len(zloc) > 1:
            raise ValueError('RTSTRUCT not registered to rectilinear coordinates')
        return (coords, zloc[0])

    def _build_contour(self, contour_data):
        fill_array = np.zeros(self.dims.shape)
        for contour_slice in contour_data:
            coords, zloc = self._slice_coords(contour_slice)

            poly2D = np.zeros(self.dims.shape[:2])
            cv2.fillPoly(poly2D, coords, 1)

            fill_array[:, :, zloc] += poly2D
        fill_array = fill_array % 2
        return fill_array

    def check_dims(self):
        def _check_dims(func):
            def wrapped(self, modality, *args, **kwargs):
                if not hasattr(self, 'dims'):
                    if modality.name in ['CT', 'MR']:
                        self.dims = VolumeDimensions(modality.data)
                func(*args, **kwargs)
            return wrapped
        return _check_dims

    def define_vol_dims(self, frame_of_ref):
        if not hasattr(self, 'dims'):
            for mod in frame_of_ref.iter_modalities:
                if mod.name in ['CT', 'MR']:
                    self.dims = VolumeDimensions(mod.data)
                    break
            if not hasattr(self, 'dims'):
                raise TypeError('Volume dimensions not created, no MR or CT in Frame Of Reference')

    def struct(self, modality):
        if not hasattr(self, 'dims'):
            raise LookupError('Must define volume dimensions first')
        # Return as a struct volume type
        struct_sets = []
        for struct_group in modality.struct:
            struct_set = StructVolumeSet(self.dims)
            for struct in struct_group:
                ds = pydicom.dcmread(struct.filepath)
                for index, contour in enumerate(ds.StructureSetROISequence):
                    name = contour.ROIName.lower()
                    contour_data = ds.ROIContourSequence[index].ContourSequence
                    built = self._build_contour(contour_data)
                    struct_set[name] = built
            struct_sets.append(struct_set)
        return struct_sets

    @check_dims
    def ct(self, modality, HU=True):
        ct_vols = []
        for ct_group in modality.ct:
            fill_array = np.zeros(self.dims.shape, dtype='float32')
            origin = self.dims.origin

            for ct_file in ct_group:
                ds = pydicom.dcmread(ct_file)
                z_loc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, z_loc] = ds.pixel_array

            slope = ds.RescaleSlope
            intercept = ds.RescaleIntercept
            if HU:
                ct_vols.append(fill_array * slope + intercept)
            else:
                ct_vols.append(fill_array)
        return ct_vols

    @check_dims
    def nm(self, modality):
        nm_vols = []
        for nm_group in modality.nm:
            ds = pydicom.dcmread(nm_group[0])
            raw = np.rollaxis(ds.pixel_array, 0, 3)[:, :, ::-1]
            map_seq = ds.RealWorldValueMappingSequence[0]
            slope = map_seq.RealWorldValueSlope
            intercept = map_seq.RealWorldValueIntercept
            nm_vols.append(raw * slope + intercept)
        return nm_vols

    @check_dims
    def mr(self, modality):
        mr_vols = []
        for mr_group in modality.mr:
            fill_array = np.zeros(mr_group.dims)
            origin = mr_group.origin

            for mr_file in mr_group:
                ds = pydicom.dcmread(mr_file)
                z_loc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, z_loc] = ds.pixel_array
            mr_vols.append(fill_array)
        return mr_vols

    @check_dims
    def pet(self, modality):
        pet_vols = []
        for pet_group in modality.pet:
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

    @check_dims
    def dose(self, modality):
        dose_vols = {}
        ct_coords = self.dims.coordgrid()
        for dosefile in modality.dose:
            ds = pydicom.dcmread(dosefile.filepath)
            dose_dims = VolumeDimensions(dosefile)
            dose_coords = dose_dims.coordrange()
            dose_array = np.rollaxis(ds.pixel_array, 0, 3) * ds.DoseGridScaling
            interper = RGI(dose_coords, dose_array,
                           bounds_error=False, fill_value=0)
            dose_interp = interper(ct_coords).reshape(self.dims.shape)
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
