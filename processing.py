import cv2
import numpy as np
import pydicom
import utils
from scipy.interpolate import RegularGridInterpolator as RGI
from dataclasses import dataclass, field, fields
from utils import VolumeDimensions, check_dims


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
        utils.colorwarn('Hello!')
        # we can multithread on this iterable
        for mod in frame_of_ref.iter_modalities:
            # TODO: Make switch statement in python 3.10
            if mod.dirname == 'CT':
                print('in ct recon')
                vol = self.ct(mod)
            """
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

            for ct_file in sorted(ct_group):
                ds = pydicom.dcmread(ct_file.filepath)
                zloc = int(round(abs(origin[-1] - ds.SliceLocation)))
                print(zloc, origin[-1], ds.SliceLocation, self.dims.zlohi)
                fill_array[:, :, zloc] = ds.pixel_array

            if HU:
                slope = ds.RescaleSlope
                intercept = ds.RescaleIntercept
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
            slope = float(map_seq.RealWorldValueSlope)
            intercept = float(map_seq.RealWorldValueIntercept)
            if intercept != 0:
                utils.colorwarn('NM intercept not zero and may be corrupted.')
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
