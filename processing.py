from __future__ import annotations
import cv2
import numpy as np
import pydicom
import groupings
import utils
from scipy.interpolate import RegularGridInterpolator as RGI
from skimage.transform import rescale
from dataclasses import dataclass
from utils import VolumeDimensions, check_dims
from deconstruction import RTStructConstructor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groupings import Cohort, FrameOfRef, Modality, ReconstructedVolume, ReconstructedFile


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
    ratio: float = None

    # interpolation
    interpolated: bool = False
    empty_slices: list = None
    interpolated_slices: list = None

    # biase field correction
    bias_corrected: bool = False

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
        # TODO: Need to update the img.VolumeDimensions to correlate too
        self.resampled = True
        self.pixelspacing_original = pixelspacing_original
        self.ratio = ratio

    def interpolated_update(self, interpolated_slices: list, extrapolated_slices: list) -> None:
        temp = self.empty_slices
        if len(interpolated_slices) > 0:
            self.interpolated = True
            self.interpolated_slices = interpolated_slices
            temp = list(set(temp) - set(interpolated_slices))
        if len(extrapolated_slices) > 0:
            self.extrapolated = True
            self.extrapolated_slices = extrapolated_slices
            temp = list(set(temp) - set(extrapolated_slices))
        self.empty_slices = temp

    def as_dict(self):
        return vars(self)

    def from_dict(self, fields):
        for name, value in fields.items():
            setattr(self, name, value)
        return self

    def export(self):
        return vars(self)


class Reconstruction:
    def __init__(self, filter_structs: list = None):
        self.filter_structs = filter_structs
        self.in_memory = True

    def __call__(self, frame_of_ref: FrameOfRef, in_memory: bool = False,
                 path: str = None) -> None:
        if not hasattr(self, 'dims'):
            self.define_vol_dims(frame_of_ref)

        self.in_memory = in_memory
        self.path = path

        all_pointers = list(map(self._split_mod, frame_of_ref.iter_modalities()))
        return all_pointers

    def _split_mod(self, mod: Modality):
        """[Wrapper function to split the modalities to their respective functions]

        Args:
            mod (Modality): [Modality to reconstruct]

        Returns:
            [tuple]: [Modality and ReconstructedFile]

        Notes:
            # Match statement in python 3.10
            match mod.name:
                case 'CT':
                    pointer = self.ct(mod)
                ...
                case _:
                    raise TypeError(f'Reconstruction of {mod.name} not supported')
        """
        if mod.name == 'CT':
            pointer = self.ct(mod)
        elif mod.name == 'RTSTRUCT':
            pointer = self.struct(mod)
        elif mod.name == 'MR':
            pointer = self.mr(mod)
        elif mod.name == 'NM':
            pointer = self.nm(mod)
        elif mod.name == 'PET':
            pointer = self.pet(mod)
        elif mod.name == 'DOSE':
            pointer = self.dose(mod)
        return (mod, pointer)

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
        fill_array = np.zeros(self.dims.shape, dtype=np.uint8)
        for contour_slice in contour_data:
            coords, zloc = self._slice_coords(contour_slice)
            poly2D = np.zeros(self.dims.shape[:2])
            cv2.fillPoly(poly2D, coords, 1)
            fill_array[..., zloc] += np.array(poly2D, dtype=np.uint8)
        fill_array = fill_array % 2
        return fill_array

    def _struct_filter_check(self, name):
        if self.filter_structs is None:  # No filter
            return (True, name)
        elif type(self.filter_structs) is list:  # list
            contains = name in self.filter_structs
            return (contains, name)
        elif type(self.filter_structs) is dict:  # dict, rename
            if name in self.filter_structs:
                return (True, name)
            for key, value in self.filter_structs.items():
                if name in value:
                    return (True, key)
            return (False, name)
        else:
            raise TypeError('filter_by[\'StructName\'] must be dict or list')

    def define_vol_dims(self, frame_of_ref: FrameOfRef) -> None:
        """[Defines the volume dimensions from a FrameOfRef for RTSTRUCT construction]

        Args:
            frame_of_ref (FrameOfRef): [Frame of Ref with CT or MR DICOMs]

        Raises:
            TypeError: [Raised if CT or MR are not within FrameOfRef]
        """
        if not hasattr(self, 'dims'):
            for mod in frame_of_ref.iter_modalities():
                if mod.name in ['CT', 'MR']:
                    self.dims = VolumeDimensions(mod.dicoms_data)
                    break
            if not hasattr(self, 'dims'):
                print(frame_of_ref.name)
                for patient in frame_of_ref:
                    print(patient.name)
                raise TypeError('Volume dimensions not created, no MR or CT in Frame Of Reference')

    def struct(self, modality: Modality) -> None:
        """[RTSTRUCT reconstruction from a modality with RTSTRUCT DicomFiles]

        Args:
            modality (Modality): [Modality with RTSTRUCT DicomFiles]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.

        Raises:
            LookupError: [Must set volume dimensions with 'define_vol_dims' first]
        """
        if not hasattr(self, 'dims'):
            raise LookupError('Must define volume dimensions first')
        # Return as a struct volume type
        for struct_group in modality.struct:
            structs = {}
            for struct in struct_group:
                ds = pydicom.dcmread(struct.filepath)
                for index, contour in enumerate(ds.StructureSetROISequence):
                    name = contour.ROIName.lower()
                    valid, name = self._struct_filter_check(name)
                    if valid:
                        contour_data = ds.ROIContourSequence[index].ContourSequence
                        built = np.array(self._build_contour(contour_data), dtype='bool')
                        structs.update({name: built})

            struct_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            struct_set.add_structs(structs)

            if not self.in_memory:
                struct_set.convert_to_pointer(path=self.path)
                return struct_set
            modality._add_file(struct_set)

    @check_dims
    def ct(self, modality: Modality, HU: bool = True) -> None:
        """[Reconstruct CT Volume from DICOMs within Modality object]

        Args:
            modality (Modality): [Modality object containing CT DICOMs]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.
        """
        for ct_group in modality.ct:
            fill_array = np.zeros(self.dims.shape, dtype='float32')

            corrupt_slices = []
            filled_zlocs = []
            for ct_file in sorted(ct_group):
                ds = pydicom.dcmread(ct_file.filepath)
                zloc = self.dims.calc_z_index(loc=ds.ImagePositionPatient[-1])
                try:
                    fill_array[:, :, zloc] = np.array(ds.pixel_array, dtype='float32')
                    filled_zlocs.append(zloc)
                except ValueError:
                    corrupt_slices.append(zloc)
                except IndexError:
                    utils.colorwarn(f'Patient {ds.PatientID}, slice {zloc} is problematic')
                else:
                    filled_zlocs.append(zloc)

            if HU:
                slope = ds.RescaleSlope
                intercept = ds.RescaleIntercept
                fill_array = np.array(fill_array * slope + intercept, dtype='float32')

            ct_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            ct_set.add_vol(ds.SOPInstanceUID, fill_array)

            empty_slices = list(set(range(self.dims.slices)) - set(filled_zlocs))
            empty_slices.sort()
            if len(empty_slices) > 0:
                message = f'Empty slices in: {ds.PatientID}, interpolation recommended'
                source = 'DICOManager/processing.py'
                utils.colorwarn(message, source)
                #utils.colorwarn(f'Empty slices in: {ds.PatientID}, interpolation recommended')
                ct_set.ImgAugmentations.empty_slices = empty_slices

            if not self.in_memory:
                ct_set.convert_to_pointer(path=self.path)
                return ct_set
            modality._add_file(ct_set)

    @check_dims
    def nm(self, modality: Modality) -> None:
        """[Reconstruct NM DICOMs within Modality object]

        Args:
            modality (Modality): [Modality object containing NM DICOM]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.
        """
        for nm_group in modality.nm:
            ds = pydicom.dcmread(nm_group.filepath)
            raw = np.rollaxis(ds.pixel_array, 0, 3)[:, :, ::-1]
            map_seq = ds.RealWorldValueMappingSequence[0]
            slope = float(map_seq.RealWorldValueSlope)
            intercept = float(map_seq.RealWorldValueIntercept)
            if intercept != 0:
                utils.colorwarn(f'NM file: {nm_group.filepath} has non-zero intercept')
            fill_array = np.array(raw * slope + intercept, dtype='float32')

            nm_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            nm_set.add_vol(ds.SOPInstanceUID, fill_array)

            if not self.in_memory:
                nm_set.convert_to_pointer(path=self.path)
                return nm_set
            modality._add_file(nm_set)

    @check_dims
    def mr(self, modality: Modality) -> None:
        """[Reconstruct MR Volume from DICOMs within Modality object]

        Args:
            modality (Modality): [Modality object containing MR DICOMs]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.
        """
        for mr_group in modality.mr:
            fill_array = np.zeros(self.dims.shape, dtype='float32')

            empty_slices = []
            for mr_file in sorted(mr_group):
                ds = pydicom.dcmread(mr_file.filepath)
                zloc = self.dims.calc_z_index(loc=ds.ImagePositionPatient[-1])
                try:
                    fill_array[:, :, zloc] = ds.pixel_array
                except ValueError:
                    empty_slices.append(zloc)

            mr_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            mr_set.add_vol(ds.SOPInstanceUID, fill_array)

            if len(empty_slices) > 0:
                empty_slices.sort()
                print(f'Patient {ds.PatientID} missing slices: {empty_slices}')
                mr_set.ImgAugmentations.empty_slices = empty_slices

            if not self.in_memory:
                mr_set.convert_to_pointer(path=self.path)
                return mr_set
            modality._add_file(mr_set)

    @check_dims
    def pet(self, modality: Modality) -> None:
        """[Reconstruct PET DICOMs within Modality object]

        Args:
            modality (Modality): [Modality object containing PET DICOM]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.
        """
        for pet_group in modality.pet:
            fill_array = np.zeros(self.dims.shape, dtype='float32')
            origin = self.dims.origin

            for pet_file in sorted(pet_group):
                ds = pydicom.dcmread(pet_file.filepath)
                zloc = int(round(abs((origin[-1]))))
                fill_array[:, :, zloc] = ds.pixel_array

            rescaled = fill_array * ds.RescaleSlope + ds.RescaleIntercept
            patient_weight = 1000 * float(ds.PatientWeight)
            radiopharm = ds.RadiopharmaceuticalInformationSequence[0]
            total_dose = float(radiopharm.RadionuclideTotalDose)
            suv_values = rescaled * patient_weight / total_dose

            pet_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            pet_set.add_vol(ds.SOPInstanceUID, suv_values)

            if not self.in_memory:
                pet_set.convert_to_pointer(path=self.path)
                return pet_set
            modality._add_file(pet_set)

    @check_dims
    def dose(self, modality: Modality) -> None:
        """[Reconstructs RTDOSE from Modality object]

        Args:
            modality (Modality): [Modality object containing RTDOSE]
            in_place (bool, optional): [Sorts into existing tree if True]. Defaults to True.
        """
        ct_coords = self.dims.coordgrid()
        for dosefile in modality.dose:
            ds = pydicom.dcmread(dosefile.filepath)
            dose_dims = VolumeDimensions(dosefile)
            dose_coords = dose_dims.coordrange()
            dose_array = np.rollaxis(ds.pixel_array, 0, 3) * ds.DoseGridScaling
            interper = RGI(dose_coords, dose_array,
                           bounds_error=False, fill_value=0)
            dose_interp = interper(ct_coords).reshape(self.dims.shape)

            dose_set = groupings.ReconstructedVolume(ds, self.dims, parent=modality)
            dose_set.add_vol(ds.SOPInstanceUID, dose_interp)

            if not self.in_memory:
                dose_set.convert_to_pointer(path=self.path)
                return dose_set
            modality._add_file(dose_set)


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
        new_rt.append_masks(masks, roi_names)

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

