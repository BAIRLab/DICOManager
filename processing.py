from __future__ import annotations
import cv2
import numpy as np
import pydicom
import utils
from copy import deepcopy
import SimpleITK as sitk
from scipy.interpolate import RegularGridInterpolator as RGI
from skimage.transform import rescale
from dataclasses import dataclass, field, fields
from utils import VolumeDimensions, check_dims
from new_deconstruction import RTStructConstructor
from typing import TYPE_CHECKING, Union

from anytree import NodeMixin

if TYPE_CHECKING:
    from groupings import Cohort, FrameOfRef, Modality, DicomFile


class TestNode(NodeMixin):
    def __init__(self, name=None, files=None, parent=None, children=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children


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

    # biase field correction
    bias_corrected: bool = True

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

    def crop_update(self, img_coords: np.ndarray, patient_coords: np.ndarray) -> None:
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
class StructVolumeSet:
    dims: VolumeDimensions
    # Named as the structures and the volumes
    volumes: dict = field(default_factory=dict)
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


class ReconstructedVolume(NodeMixin):  # Alternative to Modality
    def __init__(self, modality: str, SeriesUID: str, dims: VolumeDimensions, *args, **kwargs):
        super().__init__()
        self.modality = modality
        self.SeriesUID = SeriesUID
        self.dims = dims
        self.data = {}
        self.ImgAugmentations = ImgAugmentations()

    def __getitem__(self, name: str):
        return self.volumes[name]

    def __setitem__(self, name: str, volume: np.ndarray):
        if name in self.data:
            name = self._rename(name)
        self.data.update({name: volume})

    def __str__(self):
        middle = []
        for index, key in enumerate(self.data):
            middle.append(f' {len(self.data[key])} files')
        output = ' [' + ','.join(middle) + ' ]'
        return output

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
            if temp not in self.data:
                break
            i += 1
        return temp

    @property
    def shape(self):
        return self.dims.shape


class Reconstruction:
    def __init__(self, tree):
        self.tree = tree

    def __call__(self, frame_of_ref: FrameOfRef,
                 filter_structs: Union[list, dict] = None) -> None:
        self.filter_structs = filter_structs
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
            mod_data = self._new_mod(mod)
            print(self.tree)

            # We need each reconstruction to return a ReconstructionVolume object
            if mod.name == 'CT':
                print('ct')
                mod_data.ct = self.ct(mod)
            elif mod.name == 'RTSTRUCT':
                print('struct')
                temp = self.struct(mod)
                print(temp[0].shape)
                mod_data.struct = temp
                print(temp[0].shape)
                new = ReconstructedVolume(modality='RTSTRUCT', SeriesUID=mod.name, dims=None)
                print(type(new))
            elif mod.name == 'MR':
                print('mr')
                mod_data.mr = self.mr(mod)
            elif mod.name == 'NM':
                print('nm')
                mod_data.nm = self.nm(mod)
            elif mod.name == 'PET':
                print('pet')
                mod_data.pet = self.pet(mod)
            elif mod.name == 'DOSE':
                print('dose')
                mod_data.dose = self.dose(mod)
            print('here i am:', type(mod_data))
            self.tree._add_file(mod_data)
        #return self.tree

    def _new_mod(self, mod):
        new_mod = deepcopy(mod)
        new_mod.parent = self.tree
        new_mod.name = 'Reconstructed_' + mod.name
        new_mod.data = {}
        new_mod._child_type = ReconstructedVolume
        new_mod._organize_by = 'SeriesUID'
        print(type(new_mod))
        return new_mod

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

    def _struct_filter_check(self, name):
        if self.filter_structs is None:
            return True
        elif type(self.filter_structs) is list:
            return name in self.filter_structs
        elif type(self.filter_structs) is dict:
            if name in self.filter_structs:
                return True
            found = False
            for value in self.filter_structs.values():
                if name in value:
                    found = True
                    break
            return found
        else:
            raise TypeError('filter_structs must be dict or list')

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
                    if self._struct_filter_check(name):
                        contour_data = ds.ROIContourSequence[index].ContourSequence
                        built = self._build_contour(contour_data)
                        struct_set[name] = built
            struct_sets.append(struct_set)
        return struct_sets

    @check_dims
    def ct(self, modality: Modality, HU=True) -> list:  # Should be a dict or tree in the future...
        TestNode('testing', ['test0'], self.tree)
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
            ds = pydicom.dcmread(nm_group.filepath)
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
            fill_array = np.zeros(self.dims.shape)
            origin = self.dims.origin

            for pet_file in sorted(pet_group):
                ds = pydicom.dcmread(pet_file.filepath)
                zloc = int(round(abs((origin[-1]))))
                fill_array[:, :, zloc] = ds.pixel_array

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
    def __init__(self, tree):
        self.tree = tree

    def to_rt(self, source_rt: Modality, masks: np.ndarray,
              roi_names: list = None, mim: bool = True,
              sort: bool = True) -> pydicom.dataset.Dataset:
        """[Appends masks to a given RTSTRUCT modality object]

        Args:
            source_rt (Modality): [description]
            masks (np.ndarray): [description]
            roi_names (list, optional): [description]. Defaults to None.
            mim (bool, optional): [description]. Defaults to True.

        Returns:
            pydicom.dataset.Dataset: [description]
        """
        if not sort:
            return self.from_rt(source_rt, masks, roi_names, mim, False, True)
        temp = self.from_rt(source_rt, masks, roi_names, mim, False, True)
        self.tree._add_file(temp)
        return None

    def from_rt(self, source_rt: Modality, masks: np.ndarray,
                roi_names: list = None, mim: bool = True, empty: bool = True,
                update: bool = True, sort: bool = True) -> pydicom.dataset.Dataset:
        parent = source_rt.parent
        while not type(parent) is FrameOfRef:
            parent = parent.parent
            assert type(parent) != Cohort, 'Tree must contain FrameOfRef'
        frame_of_ref = parent

        if roi_names:
            warning = 'No names, or a name for each mask must be given'
            assert masks.shape[0] == len(roi_names), warning
        if type(source_rt) is str:
            source_rt = pydicom.dcmread(source_rt)

        new_rt = RTStructConstructor(frame_of_ref, rt_dcm=source_rt, mim=mim)
        if update:
            new_rt.update_header()
        if empty:
            new_rt.empty()

        new_rt.append_masks(masks, roi_names)

        if not sort:
            return new_rt.to_pydicom()
        temp = new_rt.to_pydicom()
        self.tree._add_file(temp)
        return None

    def from_ct(self, masks: np.ndarray,
                roi_names: list = None, mim: bool = True,
                sort: bool = True, save: bool = False) -> pydicom.dataset.Dataset:
        if roi_names:
            warning = 'No names, or a name for each mask must be given'
            assert masks.shape[0] == len(roi_names), warning

        new_rt = RTStructConstructor(self.tree, mim=mim)
        new_rt.initialize()
        print('initialized')
        new_rt.append_masks(masks, roi_names)
        print('masks added')

        if not sort:
            return new_rt.to_pydicom()
        temp = new_rt.to_pydicom()
        self.tree._add_file(temp)
        return None

    def save_rt(self, source_rt: pydicom.dataset.Dataset, filename: str = None) -> None:
        """[Save created RTSTRUCT to specified filepath]

        Args:
            source_rt (pydicom.dataset.Dataset): [RTSTRUCT to save]
            filename (str, optional): [Default name is ./SOPInstanceUID]. Defaults to None.

        Raises:
            TypeError: [Occurs if source_rt is not a complete pydicom.dataset object]
        """
        if not filename:
            try:
                filename = source_rt.SOPInstanceUID + '.dcm'
            except Exception:
                raise TypeError('source_rt must be a pydicom.dataset object')

        if type(source_rt) is pydicom.dataset.FileDataset:
            source_rt.save_as(filename)
        elif type(source_rt) is pydicom.dataset.Dataset:
            # P.10.C.7.1 DICOM File Meta Information
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.FileMetaInformationGroupLength = 222  # Check
            file_meta.FileMetaInformationVersion = b'\x00\x01'
            file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
            file_meta.MediaStorageSOPInstanceUID = source_rt.SOPInstanceUID
            file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.UID('1.2.276.0.7230010.3.0.3.6.2')
            file_meta.ImplementationVersionName = 'OFFIS_DCMTK_362'
            file_meta.SourceApplicationEntityTitle = 'RO_AE_MIM'

            # Input dict to convert pydicom Datastet to FileDataset
            inputs = {'filename_or_obj': filename,
                      'dataset': source_rt,
                      'file_meta': file_meta,
                      'preamble': b'\x00'*128}
            output_ds = pydicom.dataset.FileDataset(**inputs)
            output_ds.save_as(filename)
        else:
            raise TypeError('source_rt must be a pydicom.dataset object')

    def sort_rt(self, source_rt):
        self.tree._add_file(source_rt)


class Tools:
    # Tools are used to process reconstructed arrays
    # DicomUtils are for DICOM reconstruction utilities
    # Should probably unnest from the class and move to a seperate file
    def dose_max_points(self, dose_array: np.ndarray,
                        dose_coords: np.ndarray = None) -> np.ndarray:
        """[Calculates the dose maximum point in an array, returns index or coordinates]

        Args:
            dose_array (np.ndarray): [A reconstructed dose array]
            dose_coords (np.ndarray, optional): [Associated patient coordinates]. Defaults to None.

        Returns:
            np.ndarray: [The dose max index, or patient coordinates, if given]
        """
        index = np.unravel_index(np.argmax(dose_array), dose_array.shape)
        if dose_coords:
            return dose_coords[index]
        return index

    def window_level(self, img: ReconstructedVolume, window: int, level: int) -> ReconstructedVolume:
        """[Applies a window and level to the given object. Works for either HU or CT number,
            whichever was specified during reconstruction of the array]

        Args:
            img (ReconstructedVolume): [Image volume object to window and level]
            window (int): [window in either HU or CT number]
            level (int): [level in either HU or CT number]

        Returns:
            ReconstructedVolume: [Image volume with window and level applied]
        """
        imgmin, imgmax = (img.min(), img.max())
        img.augmetnations.wl_update(window, level, imgmin, imgmax)

        ctlo = level - window / 2
        cthi = ctlo + window

        temp = np.copy(img.array)
        temp[temp < ctlo] = ctlo
        temp[temp > cthi] = cthi
        img.array = temp
        return img

    def normalize(self, img: ReconstructedVolume) -> ReconstructedVolume:
        """[Normalizes an image volume ojbect]

        Args:
            img (ReconstructedVolume): [The image volume object to normalize]

        Returns:
            ReconstructedVolume: [Normalized image volume object]
        """
        imgmin, imgmax = (img.array.min(), img.array.max())
        img.augmentations.norm_update(imgmin, imgmax)

        temp = (np.copy(img.array) - imgmin) / (imgmax - imgmin)
        img.array = temp
        return img

    def standardize(self, img: ReconstructedVolume) -> ReconstructedVolume:
        """[Standardizes an image volume object]

        Args:
            img (ReconstructedVolume): [The image volume object to standardize]

        Returns:
            ReconstructedVolume: [Standardized image volume object]
        """
        imgmean = np.mean(img.array)
        imgstd = np.std(img.array)
        img.augmentation.std_update(imgmean, imgstd)

        temp = (np.copy(img.array) - imgmean) / imgstd
        img.array = temp
        return img

    def crop(self, img: ReconstructedVolume, centroid: np.ndarray,
             crop_size: np.ndarray) -> ReconstructedVolume:
        """[Crops an image volume and updates headers accordingly]

        Args:
            img (ReconstructedVolume): [Image volume object to crop]
            centroid (np.ndarray): [central cropping value]
            crop_size (np.ndarray): [dimensions of final cropped array]

        Returns:
            ReconstructedVolume: [description]

        Notes:
            Centroid will not be observed if the cropped volume will be
                smaller than the crop_size. Will shift centroid to maintain
                the specific crop size
            This function currently does not update the ReconstructedVolume header
                or ReconstructedVolume.dicom_header to reflect the new dimensions
        """
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

        # img.crop_update()
        img.augmentations.crop_update(img_coords, patient_coords)
        return img

    def resample(self, img: ReconstructedVolume, ratio: float) -> ReconstructedVolume:
        """[Downsample an image by a specified ratio]

        Args:
            img (ReconstructedVolume): [Image to resample]
            ratio (float): [Ratio to downsample]

        Returns:
            ReconstructedVolume: [Downsampled image array]

        Notes:
            TODO: Does not update voxel spacing yet
        """
        temp = np.copy(img.array)
        img.array = rescale(temp, ratio)
        img.augmentations.resample_update(None, round(ratio))
        return img

    def bias_field_correction(self, img: ReconstructedVolume) -> ReconstructedVolume:
        """[MRI Bias Field Correction]

        Args:
            img (ReconstructedVolume): [MRI Image Volume to be N4 bias corrected]

        Returns:
            ReconstructedVolume: [N4 bias corrected image]
        """
        img.augmentations.bias_corrected = True

        sitk_image = sitk.GetImageFromArray(img.array)
        sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output = corrector.Execute(sitk_image, sitk_mask)
        img.array = sitk.GetArrayFromImage(output)

        return img
