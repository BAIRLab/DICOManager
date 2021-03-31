import cv2
import numpy as np
import pydicom
import utils
from scipy.interpolate import RegularGridInterpolator as RGI
from dataclasses import dataclass, field, fields
from utils import VolumeDimensions, check_dims
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groupings import FrameOfRef, Modality

@dataclass
class StructVolumeSet:
    dims: VolumeDimensions
    volumes: dict = field(default_factory=dict)  # Named as the structures and the volumes
    interpoltaed: bool = False
    missing_slices: list = None

    def __getitem__(self, name: str):
        return self.volumes[name]

    def __setitem__(self, name: str, volume: np.ndarray):
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
class ImgAugmentations:
    # Window level augmentation metadata
    window_level: bool = False
    window: float = None
    level: float = None
    wl_original_range: tuple = None

    # normalization augmentation metadata
    normalized: bool = False
    norm_original_range: tuple = None

    # standardization augmetnation metadata
    standardized: bool = False
    std: float = None
    mean: float = None

    # cropping augmetnation metata
    cropped: bool = False
    img_coords: np.ndarray = None
    patient_coords: np.ndarray = None

    # resampling augmentation metadata
    resampled: bool = False
    pixelspacing_original: np.ndarray = None

    # interpolation
    interpolated: bool = False
    interpolated_slices: list = None

    def wl_update(self, window: int, level: int, imgmin: float, imgmax: float) -> None:
        self.window_level = True
        self.window = window
        self.level = level
        self.wl_original_range = (imgmin, imgmax)

    def norm_update(self, imgmin: float, imgmax: float) -> None:
        self.normalized = True
        self.norm_original_range = (imgmin, imgmax)

    def std_update(self, std: float, mean: float) -> None:
        self.standardized = True
        self.std = std
        self.mean = mean

    def crop_update(self, img_coors: np.ndarray, patient_coords: np.ndarray) -> None:
        self.cropped = True
        self.img_coords = img_coords
        self.patient_coords = patient_coords

    def resampled_update(self, pixelspacing_original: np.ndarray, ratio: int) -> None:
        self.resampled = True
        self.pixelspacing_original = pixelspacing_original
        self.ratio = ratio

    def interpolated_update(self, interpolated_slices: list) -> None:
        self.interpolated = True
        self.interpolated_slices = interpolated_slices



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
    augmentations: ImgAugmentations = field(default_factory=ImgAugmentations)


class Reconstruction:
    def __call__(self, frame_of_ref: FrameOfRef):
        if not hasattr(self, 'dims'):
            self.define_vol_dims(frame_of_ref)
        """
        # Match statement in python 3.10
        match mod.name:
            case 'CT':
                vols = self.ct(mod)
            case 'MR':
                vols = self.mr(mod)
            case 'NM':
                vols = self.nm(mod)
            case 'PET':
                vols = self.pet(mod)
            case 'DOSE':
                vols = self.dose(mod)
            case 'RTSTRUCT':
                vols = self.struct(mod)
            case _:
                raise TypeError(f'Reconstruction of {mod.name} not supported')
        """
        for mod in frame_of_ref.iter_modalities:
            if mod.name == 'CT':
                vols = self.ct(mod)
            elif mod.name == 'RTSTRUCT':
                vols = self.struct(mod)
            elif mod.name == 'MR':
                vols = self.mr(mod)
            # These might not work yet
            elif mod.name == 'NM':
                vols = self.nm(mod)
            elif mod.name == 'PET':
                vols = self.pet(mod)
            elif mod.name == 'DOSE':
                vols = self.dose(mod)
        return vols

    def _slice_coords(self, contour_slice: pydicom.dataset.Dataset) -> (np.ndarray, int):
        """[converts a dicom contour slice into image coordinates]

        Args:
            contour_slice (pydicom.dataset.Dataset): [(3006, 0016): Contour Image Sequence]

        Returns:
            [np.ndarray]: [2D numpy array of contour points in image coordinates]
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

    def _build_contour(self, contour_data: np.ndarray) -> np.ndarray:
        fill_array = np.zeros(self.dims.shape)
        for contour_slice in contour_data:
            coords, zloc = self._slice_coords(contour_slice)

            poly2D = np.zeros(self.dims.shape[:2])
            cv2.fillPoly(poly2D, coords, 1)

            fill_array[:, :, zloc] += poly2D
        fill_array = fill_array % 2
        return fill_array

    def define_vol_dims(self, frame_of_ref: FrameOfRef) -> None:
        if not hasattr(self, 'dims'):
            for mod in frame_of_ref.iter_modalities:
                if mod.name in ['CT', 'MR']:
                    self.dims = VolumeDimensions(mod.data)
                    break
            if not hasattr(self, 'dims'):
                raise TypeError('Volume dimensions not created, no MR or CT in Frame Of Reference')

    def struct(self, modality: Modality):
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
    def ct(self, modality: Modality, HU=True) -> list:  # Should be a dict or tree in the future...
        ct_vols = []
        for ct_group in modality.ct:
            fill_array = np.zeros(self.dims.shape, dtype='float32')
            origin = self.dims.origin

            for ct_file in sorted(ct_group):
                ds = pydicom.dcmread(ct_file.filepath)
                zloc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, zloc] = ds.pixel_array

            if HU:
                slope = ds.RescaleSlope
                intercept = ds.RescaleIntercept
                ct_vols.append(fill_array * slope + intercept)
            else:
                ct_vols.append(fill_array)
        return ct_vols

    @check_dims
    def nm(self, modality: Modality) -> list:
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
    def mr(self, modality: Modality) -> list:
        mr_vols = []
        for mr_group in modality.mr:
            fill_array = np.zeros(self.dims.shape, dtype='float32')
            origin = self.dims.origin

            for mr_file in sorted(mr_group):
                ds = pydicom.dcmread(mr_file.filepath)
                zloc = int(round(abs(origin[-1] - ds.SliceLocation)))
                fill_array[:, :, zloc] = ds.pixel_array
            mr_vols.append(fill_array)
        return mr_vols

    @check_dims
    def pet(self, modality: Modality) -> list:
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
    def dose(self, modality: Modality) -> list:
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
    def dose_max_points(self, dose_array: np.ndarray, dose_coords: np.ndarray = None):
        index = np.unravel_index(np.argmax(dose_array), dose_array.shape)
        if dose_coords:
            return dose_coords[index]
        return index

    def window_level(self, img: ImageVolume, window: int, level: int) -> ImageVolume:
        imgmin, imgmax = (img.min(), img.max())
        img.augmetnations.wl_update(window, level, imgmin, imgmax)

        ctlo = level - window / 2
        cthi = ctlo + window

        temp = np.copy(img.array)
        temp[temp < ctlo] = ctlo
        temp[temp > cthi] = cthi
        img.array = temp
        return img

    def normalize(self, img: ImageVolume) -> ImageVolume:
        imgmin, imgmax = (img.min(), img.max())
        img.augmentations.norm_update(imgmin, imgmax)

        temp = (np.copy(img.array) - imgmin) / (imgmax - imgmin)
        img.array = temp
        return img

    def crop(self, img: ImageVolume, centroid: np.ndarray,
             crop_size: np.ndarray) -> ImageVolume:
        # Need to update the VolumeDimensions header and dicom header too
        imgshape = img.array.shape()
        img_coords = np.zeros((2, len(imgshape)))
        patient_coords = np.zeros((2, len(imgshape)))

        for i, (point, size) in enumerate(zip(centroid, crop_size)):
            low = min(0, point - size // 2)
            high = low + size
            if high > (imgshape[i] - 1):
                high = (imgshape[i] - 1)
                low = high - size
            img_coords[0, i] = low
            img_coords[1, i] = high

        coordrange = img.dims.coordrange()
        for i, (low, high) in enumerate(img_coords.T):
            patient_coords[0, i] = coordrange[i, low]
            patient_coords[1, i] = coordrange[i, high]

        xlo, xhi, ylo, yhi, zlo, zhi = img_coords.T.flatten()
        temp = np.copy(img.array)
        img.array = temp[xlo: xhi, ylo: yhi, zlo: zhi]

        #img.crop_update()
        img.augmentations.crop_update(img_coords, patient_coords)
        return img


# Can reconstruct either via Series.recon.ct()
# or processing.recon.ct()
# This will require us to design the file structure accordingly
