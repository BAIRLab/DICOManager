#! /usr/bin/python
import copy
import pydicom
import utils
from datetime import datetime
from dataclasses import dataclass
from matplotlib import cm
import numpy as np
import hashlib
import os
import random
import uuid
import scipy
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy import spatial
from skimage import measure as skm
from utils import VolumeDimensions


class RTStructConstructor:
    """
    Attributes
    ----------
    ct_series : list
        A list of posix paths to the DICOMs for a CT image volume
    rt_dcm : pydicom.dataset.FileDataset
        The current working pydicom.dataset.Filedataset
        A deep copy is made if an RTSTRUCT is provided
    mim : bool (Default = True)
        Creates MIM style contours if True
            MIM connects holes with a line of width zero
            MIM adheres to the DICOM C.8.8.6.3 standard for inner
                and outer contours
        Creates Pinnacle style contours if False
            Pinnacle creates separate structures for inner and outer contours

    Methods
    ----------
    initialize : (None) -> None
        Creates a new rtstruct dicom file from the provided ct_series
    empty : (None) -> None
        Removes all contour data from the provided rt_dcm file
    update_header: (None) -> None
        Updates new_rt header to represent a unique RTSTRUCT file
    to_pydicom : (None) -> pydicom.dataset.FileDataset
        Returns a pydicom.dataset.FileDataset object
    append_masks: (masks:np.ndarray, roi_names=None:[str]) -> None
        Appends the contours and ROI names to new_rt

    Notes
    ----------
    Helper functions for this class are avaliable:
        deconstruction.to_rt(...): append a mask to a provided RTSTRUCT
        deconstruction.from_rt(...): create a new file from an RTSTRUCT
        deconstruction.from_ct(...): create a new file from an CT image
        deconstruction.save_rt(...): save the generated pydicom.FileDataset
    """

    def __init__(self, frame_of_ref, rt_dcm=None, mim=True):
        self.frame_of_ref = frame_of_ref
        self.new_rt = copy.deepcopy(rt_dcm)
        self.mim = mim
        self._define_vol_dims()
        self._ref_sop_uids = {}
        self._ct_series_hdrs = {}
        self._unpack_ct_hdrs()

    def _define_vol_dims(self):
        for mod in self.frame_of_ref.iter_modalities:
            print(mod, mod.name)
            if mod.name == 'CT':
                self.dims = VolumeDimensions(mod.data)
                self.ct_series = [f.filepath for f in mod.ct[0]]
                break
        if not hasattr(self, 'dims'):
            raise TypeError('Volume dimensions not created, no CT in Frame Of Reference')

    def initialize(self):
        """
        Function
        ----------
        Initializes a newly registered DICOM RTSTRUCT from a CT image

        Parameters
        ----------
        None

        Modifies
        ----------
        new_rt : pydicom.dataset.FileDataset
            Initializes the currently working RTSTRUCT file to become a
            pydicom.dataset.FileDataset, which is stored at self.new_rt

        Notes
        ----------
        Modules referenced throughout code based on DICOM standard layout:
            http://dicom.nema.org/medical/dicom/current/output/html/part03.html
        In-line references follow the format of Part (P) and Chapter (C) as P.#.C.#
        Common names references are provided with the byte location (XXXX, XXXX),
            and therein referred to with the byte locations
        """
        # Read the ct to build the header from
        ct_dcm = pydicom.dcmread(self.ct_series[0], stop_before_pixels=True)

        # Start crafting a fresh RTSTRUCT
        rt_dcm = pydicom.dataset.Dataset()

        date = datetime.now().date().strftime('%Y%m%d')
        time = datetime.now().time().strftime('%H%M%S.%f')[:-3]
        instance_uid = generate_instance_uid()

        # P.3.C.7.1.1 Patient Module
        if hasattr(ct_dcm, 'PatientName'):
            rt_dcm.PatientName = ct_dcm.PatientName
        else:
            rt_dcm.PatientName = 'UNKNOWN^UNKNOWN^^'

        if hasattr(ct_dcm, 'PatientID'):
            rt_dcm.PatientID = ct_dcm.PatientID
        else:
            rt_dcm.PatientID = '0000000'

        if hasattr(ct_dcm, 'PatientBirthDate'):
            rt_dcm.PatientBirthDate = ct_dcm.PatientBirthDate
        else:
            rt_dcm.PatientBirthDate = ''

        if hasattr(ct_dcm, 'PatientSex'):
            rt_dcm.PatientSex = ct_dcm.PatientSex
        else:
            rt_dcm.PatientSex = ''

        if hasattr(ct_dcm, 'PatientAge'):
            rt_dcm.PatientAge = ct_dcm.PatientAge
        else:
            rt_dcm.PatientAge = ''

        # P.3.C.7.2.1 General Study Module
        rt_dcm.StudyInstanceUID = ct_dcm.StudyInstanceUID

        if hasattr(ct_dcm, 'StudyDate'):
            rt_dcm.StudyDate = ct_dcm.StudyDate
        else:
            rt_dcm.StudyDate = date

        if hasattr(ct_dcm, 'StudyTime'):
            rt_dcm.StudyTime = ct_dcm.StudyTime
        else:
            rt_dcm.StudyTime = time

        if hasattr(ct_dcm, 'StudyID'):
            rt_dcm.StudyID = ct_dcm.StudyID

        if hasattr(ct_dcm, 'StudyDescription'):
            rt_dcm.StudyDescription = ct_dcm.StudyDescription

        # P.3.C.7.5.1 General Equipment Module
        rt_dcm.Manufacturer = 'Beaumont Artificial Intelligence Lab'
        rt_dcm.InstitutionName = 'Beaumont Health'
        rt_dcm.ManufacturersModelName = 'DICOManager'
        rt_dcm.SoftwareVersions = ['0.1.0']

        # P.3.C.8.8.1 RT Series Module
        rt_dcm.Modality = 'RTSTRUCT'
        rt_dcm.SeriesInstanceUID = instance_uid

        if hasattr(ct_dcm, 'SeriesNumber'):
            rt_dcm.SeriesNumber = ct_dcm.SeriesNumber

        rt_dcm.SeriesDate = date
        rt_dcm.SeriesTime = time

        if hasattr(ct_dcm, 'SeriesDescription'):
            rt_dcm.SeriesDescription = ct_dcm.SeriesDescription

        # P.3.C.8.8.5 Structure Set Module
        rt_dcm.StructureSetLabel = 'Auto-Segmented Contours'
        rt_dcm.StructureSetName = 'Auto-Segmented Contours'
        rt_dcm.StructureSetDescription = 'Auto-Segmented Contours'
        rt_dcm.StructureSetDate = date
        rt_dcm.StructureSetTime = time

        # Referenced frame of reference (3006,0009)
        ref_frame_of_ref_ds = pydicom.dataset.Dataset()
        ref_frame_of_ref_ds.FrameOfReferenceUID = ct_dcm.FrameOfReferenceUID

        # RT Referenced Series Sequence (3006,0012)
        rt_ref_study_ds = pydicom.dataset.Dataset()
        # TODO: The un-commented UID is Retired, MIM uses that one
        # rt_ref_study_ds.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.3.1.2.3.2')
        rt_ref_study_ds.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.3.1.2.3.1')
        rt_ref_study_ds.ReferencedSOPInstanceUID = ct_dcm.StudyInstanceUID

        # RT Referenced Series Sequence (3006,0014)
        rt_ref_series_ds = pydicom.dataset.Dataset()
        rt_ref_series_ds.SeriesInstanceUID = ct_dcm.SeriesInstanceUID
        rt_ref_series_ds.ContourImageSequence = pydicom.sequence.Sequence(self._contour_img_seq())

        # (3006,0014) attribute of (3006,0012)
        rt_ref_study_ds.RTReferencedSeriesSequence = pydicom.sequence.Sequence([rt_ref_series_ds])

        # (3006,0012) attribute of (3006,0009)
        ref_frame_of_ref_ds.RTReferencedStudySequence = pydicom.sequence.Sequence([rt_ref_study_ds])

        # (3006,0009) attribute of (3006,0010)
        rt_dcm.ReferencedFrameOfReferenceSequence = pydicom.sequence.Sequence([ref_frame_of_ref_ds])

        # Structure Set Module (3006, 0020)
        rt_dcm.StructureSetROISequence = pydicom.sequence.Sequence()

        # P.3.C.8.8.6 ROI Contour Module
        rt_dcm.ROIContourSequence = pydicom.sequence.Sequence()

        # P.3.C.8.8.8 RT ROI Observation Module
        rt_dcm.RTROIObservationsSequence = pydicom.sequence.Sequence()

        # P.3.C.12.1 SOP Common Module Attributes
        rt_dcm.SOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
        rt_dcm.SOPInstanceUID = instance_uid
        rt_dcm.InstanceCreationDate = date
        rt_dcm.InstanceCreationTime = time

        # P.10.C.7.1 DICOM File Meta Information
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 222
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
        file_meta.MediaStorageSOPInstanceUID = instance_uid
        file_meta.ImplementationClassUID = pydicom.uid.UID('1.2.276.0.7230010.3.0.3.6.2')
        file_meta.ImplementationVersionName = 'OFFIS_DCMTK_362'
        file_meta.SourceApplicationEntityTitle = 'RO_AE_MIM'

        # Make a pydicom.dataset.Dataset() into a pydicom.dataset.FileDataset()
        inputs = {'filename_or_obj': None,
                  'dataset': rt_dcm,
                  'file_meta': file_meta,
                  'preamble': b'\x00'*128}
        self.new_rt = pydicom.dataset.FileDataset(**inputs)

    def empty(self):
        """
        Function
        ----------
        Given an RTSTRUCT pydicom object, all contours are removed

        Parameters
        ----------
        None

        Modifies
        ----------
        new_rt : pydicom.dataset.FileDataset
            Clears out contour information from the currently working RTSTRUCT
        """
        assert self.new_rt is not None, 'No RTSTRUCT provided or initialized'

        self.new_rt.StructureSetROISequence.clear()
        self.new_rt.ROIContourSequence.clear()
        self.new_rt.RTROIObservationsSequence.clear()

    def update_header(self):
        """
        Function
        ----------
        Given an RTSTRUCT pydicom object, times and UIDs are updated

        Parameters
        ----------
        None

        Modifies
        ----------
        new_rt : pydicom.dataset.FileDataset
            Updates the header of the currently working RTSTRUCT
        """
        assert self.new_rt is not None, 'No RTSTRUCT provided or initialized'

        date = datetime.now().date().strftime('%Y%m%d')
        time = datetime.now().time().strftime('%H%M%S.%f')[:-3]
        instance_uid = generate_instance_uid()

        # P.3.C.7.5.1 General Equipment Module
        self.new_rt.Manufacturer = 'Beaumont Health'
        self.new_rt.InstitutionName = 'Beaumont Health'
        self.new_rt.ManufacturersModelName = 'DICOManager'
        self.new_rt.SoftwareVersions = ['0.1.0']

        # P.3.C.8.8.1 RT Series Module
        self.new_rt.SeriesInstanceUID = instance_uid
        self.new_rt.SeriesDate = date
        self.new_rt.SeriesTime = time

        # P.3.C.8.8.5 Structure Set Module
        self.new_rt.StructureSetLabel = 'Auto-Segmented Contours'
        self.new_rt.StructureSetName = 'Auto-Segmented Contours'
        self.new_rt.StructureSetDate = date
        self.new_rt.StructureSetTime = time

        # P.3.C.12.1 SOP Common Module Attributes
        self.new_rt.SOPInstanceUID = instance_uid
        self.new_rt.InstanceCreationDate = date
        self.new_rt.InstanceCreationTime = time

        # P.10.C.7.1 DICOM File Meta Information
        self.new_rt.file_meta.MediaStorageSOPInstanceUID = instance_uid

    def append_masks(self, masks, roi_names=None):
        """
        Function
        ----------
        Appends the provided contours onto the currently working RTSTRUCT

        Parameters
        ----------
        masks : numpy.ndarray
            A boolean array of N segmentation masks with dimension [N, x, y, z]
        roi_names : [str] (Default = None)
            A list of N strings, representing the names of each contour

        Modifies
        ----------
        new_rt : pydicom.dataset.FileDataset
            Appends the contours to the current working RTSTRUCT
        """
        assert self.new_rt is not None, 'No RTSTRUCT provided or initialized'

        if not roi_names:
            roi_names = ['NewName' + str(x) for x in range(masks.shape[0])]

        n_masks = masks.shape[0]
        raw_colors = cm.rainbow(np.linspace(0, 1, n_masks))
        rgb_colors = np.array(np.round(255 * raw_colors[:, :3]), dtype='uint8')

        for i in range(n_masks):
            self._append_one_mask(masks[i], roi_names[i], rgb_colors[i])

    def to_pydicom(self):
        """
        Function
        ----------
        Returns the current working RTSTRUCT

        Returns
        ----------
        pydicom.dataset.[FileDataset, Dataset]
            The currently working RTSTRUCT file as a pydicom.datastet object
        """
        assert self.new_rt is not None, 'No RTSTRUCT provided or initialized'

        return self.new_rt

    def _append_one_mask(self, mask, roi_name, rgb_color):
        """
        Function
        ----------
        Appends coordinates and metadata into a pre-built DICOM RTSTRUCT

        Parameters
        ----------
        source_rt : pydicom.dataset.FileDataset
            An RTSTRUCT pydicom dataset object
        mask : numpy.ndarray
            A 3D boolean numpy array representing an single segmentation
        roi_name : str
            The name of the ROI to be saved in the RTSTRUCT
        rgb_color : [uint8, uint8, uint8]
            A uint8 RGB color code to encode the contour value

        Modifies
        ----------
        new_rt : pydicom.dataset.FileDataset
            The provided pydicom object is returned with appended contour

        References
        ----------
        Citations are to the DICOM NEMA standard:
            http://dicom.nema.org/medical/dicom/current/output/html/part03.html
        """
        roi_number = len(self.new_rt.StructureSetROISequence) + 1

        # P.3.C.8.8.5 Structure Set Module
        str_set_roi = pydicom.dataset.Dataset()
        str_set_roi.ROINumber = roi_number
        ref_for_seq = self.new_rt.ReferencedFrameOfReferenceSequence[0]
        str_set_roi.ReferencedFrameOfReferenceUID = ref_for_seq.FrameOfReferenceUID
        str_set_roi.ROIName = roi_name
        str_set_roi.StructureSetROIDescription = ''
        str_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        self.new_rt.StructureSetROISequence.append(str_set_roi)

        # P.3.C.8.8.6 ROI Contour Module
        roi_contour_seq = pydicom.dataset.Dataset()
        roi_contour_seq.ROIDisplayColor = list(rgb_color)
        roi_contour_seq.ReferencedROINumber = roi_number
        roi_contour_seq.ContourSequence = pydicom.sequence.Sequence([])

        # For RTSTRURCTs, a contour sequence item is a unconnected 2D z-axis polygon
        for z_index in range(mask.shape[-1]):
            z_slice = mask[:, :, z_index]
            each_polygon = separate_polygons(z_slice, mim=self.mim)
            for polygon in each_polygon:
                contour_seq = self._contour_seq(polygon, z_index)
                roi_contour_seq.ContourSequence.append(contour_seq)
        # Append entire sequence for the given contour
        self.new_rt.ROIContourSequence.append(roi_contour_seq)

        # P.3.C.8.8.8 RT ROI Observation Module
        rt_roi_obs = pydicom.dataset.Dataset()
        # This might be different than roi_number...
        rt_roi_obs.ObservationNumber = roi_number
        rt_roi_obs.ReferencedROINumber = roi_number
        rt_roi_obs.ROIObservationDescription = 'Type:Soft, Range:*/*, Fill:0, Opacity:0.0, Thickness:1, LineThickness:2, read-only:false'
        rt_roi_obs.ROIObservationLabel = roi_name
        rt_roi_obs.RTROIInterpretedType = ''
        rt_roi_obs.InterpreterType = ''
        self.new_rt.RTROIObservationsSequence.append(rt_roi_obs)

    def _contour_seq(self, polygon, z_index):
        """
        Function
        ----------
        Given a polygon and z-axis index, creates a contour sequence
            dataset object

        Parameters
        ----------
        polygon : numpy.ndarray
            A 2D numpy boolean segmentation array of a unique polygon
        z_index : int
            The index of the z-axis in the image coordinate system

        Returns
        ----------
        pydicom.dataset.Dataset
            A pydicom dataset populated with the Contour Sequence information
        """
        # Get referenced SOP UIDs and Polygon surface points
        ref_uid = self._ref_sop_uids[z_index].ref_uid
        ctcoord = self.dims.Mgrid(self._ct_series_hdrs[z_index])[..., :3]
        coords = poly_to_coords_2D(polygon, ctcoord=ctcoord)
        # Fill the Contour Sequence
        contour_seq = pydicom.dataset.Dataset()
        contour_seq.ContourImageSequence = pydicom.sequence.Sequence([ref_uid])
        contour_seq.ContourGeometricType = 'CLOSED_PLANAR'
        contour_seq.NumberOfContourPoints = len(coords) // 3
        contour_seq.ContourData = [f'{pt:0.3f}' for pt in coords]
        return contour_seq

    def _contour_img_seq(self):
        """
        Function
        ----------
        A function which organizes and sorts the image set Referenced SOP UIDs

        Parameters
        ----------
        None

        Returns
        ----------
        contour_img_seq : list
            A list of Referenced SOP Class and Instace UIDs for the image set

        Notes
        ----------
        Where _ref_sop_uids.ref_uid contains Contour Image Sequence (3006,0016)
            (3006,0016) contains Ref. SOP Class UID (0008,1150)
            (3006,0016) contian Ref. SOP Instance UID (0008,1155)
        """
        uids = [x.ref_uid for x in self._ref_sop_uids.values()]
        z_indices = list(self._ref_sop_uids)
        contour_img_seq = [x for _, x in sorted(zip(z_indices, uids))][::-1]
        return contour_img_seq

    def _unpack_ct_hdrs(self):
        """
        Function
        ----------
        Generates two dictionaries for the uid structure and ct header. Both
            dictionaries are keyed to the image coordinate z-slice location

        Parameters
        ----------
        ct_series : [str]
            A list of absolute paths to each CT DICOM file

        Modifies
        ----------
        self._ref_sop_dict
            Keyed to z-locations with values of UidItem
        self._ct_hdr_dict
            Keyed to z-locations with values of each CT .dcm header

        Notes
        ----------
        Not all the DICOM header is necessary down-stream. Memory footprint
            could be reduced by removing private tags and other info
        """
        for ct in self.ct_series:
            ct_hdr = pydicom.dcmread(ct, stop_before_pixels=True)
            parsed_uids = ParseSOPUID(ct_hdr, self.dims.dz, self.dims.zlohi[0])
            self._ref_sop_uids.update({parsed_uids.z_loc: parsed_uids})
            self._ct_series_hdrs.update({parsed_uids.z_loc: ct_hdr})


@dataclass
class ParseSOPUID:
    """
    Function
    ----------
    A dataclass for collecting and storing UID information

    Parameters
    ----------
    ct_hdr : pydicom.dataset.Dataset
        A pydicom dataset object for the CT slice DICOM object
    """
    ct_hdr: pydicom.dataset.Dataset
    ct_thick: float
    ct_loc0: float
    uid: str = None
    z_loc: int = None
    ref_uid: pydicom.dataset.Dataset = None

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def _get_z_loc(self):
        diff = self.ct_loc0 - self.ct_hdr.SliceLocation
        return round(abs(diff / self.ct_thick))

    def __post_init__(self):
        self.z_loc = self._get_z_loc()
        self.uid = self.ct_hdr.SOPInstanceUID
        self.ref_uid = pydicom.dataset.Dataset()
        self.ref_uid.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')
        self.ref_uid.ReferencedSOPInstanceUID = self.ct_hdr.SOPInstanceUID
        self.ct_hdr = None  # Clear out the header because its not needed


def to_rt(frame_of_ref, source_rt, masks, roi_names=None, mim=True):
    """
    Function
    ----------
    Adds numpy boolean array contours to a DICOM RTSTRUCT without
        modifying the existing data, but updating the SOP Instance

    Parameters
    ----------
    source_rt : str OR pydicom.dataset.FileDataset
        An absolute path to a DICOM RTSTRUCT file OR a pydicom filedataset,
            for which the new RTSTRUCT DICOM will be based
    ct_series : [str]
        A list of absolute paths to each DICOM CT file
    masks : numpy.ndarray
        A boolean numpy.ndarray of segmentation masks to be added to
            the DICOM RTSTRUCT file of dims [N, X, Y, Z] where N is
            the number of contours to be added
    roi_names : list (Default = None)
        A list of N names corresponding to each added structure. If
            not provided, the ROIs are named 'GeneratedContour#'
    mim : bool (Default = True)
        Creates MIM style contours if True
            MIM connects holes with a line of width zero
        Creates Pinnacle style contours if False
            Pinnacle creates holes as a separate structure

    Returns
    ----------
    source_rt : pydicom.dataset.FileDataset
        A RTSTRUCT pydicom dataset object for the new object

    Notes
    ----------
    This function is the same as from_rt except it does not empty the
        rtstruct. To not modify the SOP Instance either, use from_rt.
    WARNING: Identical SOP Instances associated with diffrent files
        can cause issue with certain databases
    """
    return from_rt(frame_of_ref, source_rt, masks, roi_names, mim, False, True)


def from_rt(frame_of_ref, source_rt, masks, roi_names=None,
            mim=True, empty=True, update=True):
    """
    Function
    ----------
    Creates an empty RTSTRUCT and adds a numpy boolean array contours to it

    Parameters
    ----------
    source_rt : str OR pydicom.dataset.FileDataset
        An absolute path to a DICOM RTSTRUCT file OR a pydicom filedataset,
            for which the new RTSTRUCT DICOM will be based
    ct_series : [str]
        A list of absolute paths to each DICOM CT file
    masks : numpy.ndarray
        A boolean numpy.ndarray of segmentation masks to be added to
            the DICOM RTSTRUCT file of dims [N, X, Y, Z] where N is
            the number of contours to be added
    roi_name_list : list (Default = None)
        A list of N names corresponding to each added structure. If
            not provided, the ROIs are named 'GeneratedContour#'
    mim : bool (Default = True)
        Creates MIM style contours if True
            MIM connects holes with a line of width zero
        Creates Pinnacle style contours if False
            Pinnacle creates holes as a separate structure
    empty : bool (Default = True)
        Remove all ROI data from the RTSTRUCT before appending new data

    Returns
    ----------
    source_rt : pydicom.dataset.FileDataset
        A RTSTRUCT pydicom dataset object for the new object
    """
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
    return new_rt.to_pydicom()


def from_ct(frame_of_ref, masks, roi_names=None, mim=True):
    """
    Function
    ----------
    Generates an entirely new RTSTRUCT DICOM file and saves the contours

    Parameters
    ----------
    ct_series : list
        A list of absolute paths to each DICOM CT file
    masks : numpy.ndarray
        A boolean numpy.ndarray of segmentation masks to be added to
            the DICOM RTSTRUCT file of dims [N, X, Y, Z] where N is
            the number of contours to be added
    roi_names : list (Default = None)
        A list of N names corresponding to each added structure. If
            not provided, the ROIs are named 'GeneratedContour#'
    mim : bool (Default = True)
        Creates MIM style contours if True
            MIM connects holes with a line of width zero
        Creates Pinnacle style contours if False
            Pinnacle creates holes as a separate structure

    Returns
    ----------
    source_rt : pydicom.dataset.FileDataset
        A RTSTRUCT pydicom dataset object for the new object

    Notes
    ----------
    This has not been tested with Eclipse
    """
    if roi_names:
        warning = 'No names, or a name for each mask must be given'
        assert masks.shape[0] == len(roi_names), warning

    new_rt = RTStructConstructor(frame_of_ref, mim=mim)
    new_rt.initialize()
    new_rt.append_masks(masks, roi_names)
    return new_rt.to_pydicom()


def save_rt(source_rt, filename=None):
    """
    Function
    ----------
    Given an RTSTRUCT pydicom object, save as filename. Adds necessary
        metadata if pydicom.dataset.Dataset is provided.

    Parameters
    ----------
    source_rt : pydicom.dataset.[FileDataset, Dataset]
        An RTSTRUCT pydicom dataset object to be saved
    filename : string OR posixpath (Default=None)
        A file name string or filepath to save location
        If not provided, the SOP Instance UID will be used

    Raises
    ----------
    TypeError
        Raised if input type is not pydicom.dataset.[FileDataset, Dataset]

    Notes
    ----------
    This has been seperated from the creation functions, to prevent the
        inadvertent overwriting of the original RTSTRUCT files
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
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID(
            '1.2.840.10008.5.1.4.1.1.481.3')
        file_meta.MediaStorageSOPInstanceUID = source_rt.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.UID(
            '1.2.276.0.7230010.3.0.3.6.2')
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


def wire_mask(arr: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Function
    ----------
    Given an 2D boolean array, returns those pixels on the surface

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D array corresponding to the segmentation mask
    invert : boolean (Default = False)
        Designates if the polygon is a subtractive structure from
            a larger polygon, warranting XOR dilation instead
        True when the structure is a hole / subtractive
        False when the array is a filled polygon

    Returns
    ----------
    numpy.ndarray
        A 2D boolean array, with points representing the contour surface

    Notes
    ----------
    The resultant surface is not the minimum points to fully enclose the
        surface, like the standard DICOM RTSTRUCT uses. Instead this gives
        all points along the surface. Should construct the same, but this
        calculation is faster than using Shapely.
    Could also use skimage.measure.approximate_polygon(arr, 0.001)
        To compute an approximate polygon. Better yet, do it after doing
        the binary erosion technique, and compare that to the speed of
        using shapely to find the surface
    The most accurate way would be with ITK using:
        itk.GetImageFromArray
        itk.ContourExtractor2DImageFilter
    Another option is to use VTK:
        https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/
    Either way, the reconstruction polygon fill is first priority, and then
        this deconstruction minium polygon surface computation.
    """
    assert arr.ndim == 2, 'The input boolean mask is not 2D'

    if arr.dtype != 'bool':
        arr = np.array(arr, dtype='bool')
    if invert:
        return binary_dilation(arr) ^ arr
    return binary_erosion(arr) ^ arr


def kd_sort_nearest(points: np.ndarray) -> np.ndarray:
    """
    Function
    ----------
    Given the nonzero surface points of the mask, as sorted
        array of points is returned

    Parameters
    ----------
    points : numpy.ndarray
        A Nx3 numpy array of (x, y, z) coordinates

    Returns
    ----------
    sorted_points : numpy.ndarray
        A sorted array of dimensions equal to points

    Notes
    ----------
    This method uses KDTrees which is approximately 5x slower than
        clockwise sorting, but robust to convexities
    """
    tracker = []
    sorted_points = []

    first_index = get_first_index(points)
    current = points[first_index]

    sorted_points.append(current)
    tracker.append(first_index)

    tree = spatial.cKDTree(points, balanced_tree=True)
    n_search = 2
    while len(tracker) != points.shape[0]:
        _, nearest = tree.query(current, k=n_search)

        # Expand search width if needed
        if set(nearest).issubset(set(tracker)):
            n_search += 2
            continue

        for i in nearest:
            if i not in tracker:
                current = points[i]
                sorted_points.append(current)
                tracker.append(i)
                break
    return np.array(sorted_points)


def get_first_index(points: np.ndarray) -> int:
    """
    Function
    ----------
    Returns the first index to start the kd sorting

    Parameters
    ----------
    points : numpy.ndarray
        A Nx3 numpy array of (x, y, z) coordinates

    Returns
    ----------
    index : int
        The index of the lowest point along the central line

    Notes
    ----------
    Semi-redundant with rotational sorting, but this runs
        quicker than first clockwise sorting all points
    """
    if points.dtype != 'float64':
        points = np.array(points, dtype=np.float)

    CoM = np.mean(points, axis=0)
    v1 = points - CoM
    v2 = np.array([0, -CoM[1], CoM[2]])
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    angles = np.arccos(np.dot(v1_u, v2_u))
    return np.argmin(angles)


def find_nearest(inner: np.ndarray, outer: np.ndarray) -> ([tuple, tuple], int):
    """
    Function
    ----------
    Given two arrays of polygon points, the nearest points between the
        arrays is returned

    Parameters
    ----------
    inner : numpy.ndarray OR [numpy.ndarray]
        A single numpy array with dimensions Nx3 representing the points
        for a single inner polygon or hole
    outer : numpy.ndarray
        A single numpy array with dimensions Nx3 representing the points
        for the outer polygon

    Returns
    ----------
    ([point, point], index)
        Where point is an image coordinate point, index is the index of the
        nearest element to the outer polygon

    Notes
    ----------
    This currently creates issues and needs to be fixed to work with multiple
        holes within a single polygon
    """
    outer_tree = spatial.cKDTree(outer, balanced_tree=True)
    min_dist = np.inf
    point_pair = None  # inner, outer
    index_pair = None  # inner, outer
    for i_idx, i_pt in enumerate(inner):
        dist, o_idx = outer_tree.query(i_pt)
        if dist < min_dist:
            min_dist = dist
            point_pair = [i_pt, outer[o_idx]]
            index_pair = [i_idx, o_idx]
    return (point_pair, index_pair)


def merge_sorted_points(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """
    Function
    ----------
    Given two arrays of points, a single merged list is returned

    Parameters
    ----------
    inner : numpy.ndarray OR [numpy.ndarray]
        A single numpy array or a list of numpy arrays with dimensions
        Nx3 representing the points for each inner hole or polygon
    outer : numpy.ndarray
        A single numpy array with dimensions Nx3 representing the points
        for the outer polygon

    Returns
    ----------
    numpy.ndarray
        The merged list of inner and outer polygons

    Notes
    ----------
    Because we append to merged for each inner, when larger we have iterated
        through all the points, unless the holes constitute a single pixel,
        in which they are likely undesired anyways

    Future Work
    ----------
    Could reduce ops by n points in outer if passed in outer_tree from
        when tree was sorted.
    og_outer exists to prevent inner structures from mapping to other
        inner structures which were previously inserted
    offset allows for compensation of insertion from og_outer to outer
    """
    if type(inner) is list:
        og_outer = outer.copy()
        previous = []  # for index offset due to previous insertions
        for n_inner in inner:
            point_pair, index_pair = find_nearest(n_inner, og_outer)
            n_inner = np.roll(
                n_inner, n_inner.shape[0] - index_pair[0], axis=0)
            n_inner = np.append(n_inner, point_pair, axis=0)
            offset = calc_offset(previous, index_pair[1])
            outer = np.insert(
                outer, index_pair[1] + offset + 1, n_inner, axis=0)
            previous.append((index_pair[1], n_inner.shape[0]))
    else:
        point_pair, index_pair = find_nearest(inner, outer)
        inner = np.roll(inner, inner.shape[0] - index_pair[0], axis=0)
        inner = np.append(inner, point_pair, axis=0)
        outer = np.insert(outer, index_pair[1] + 1, inner, axis=0)
    return outer


def calc_offset(previous: tuple, index: int) -> int:
    """
    Function
    ----------
    Given the previous insertions and the current insertion index,
        the necessary offset is calculated

    Paramters
    ----------
    previous : [(int, int),]
        A list of tuples with:
            tuple[0] : previous insertion index
            tuple[1] : length of previous insertion
    index : int
        The location of the current insertion into the list

    Returns
    ----------
    int
        An integer representing the number of slices to offset the insertion

    Notes
    ----------
    When we have multiple nested structures, we need to compute what the
        potential offset is when placing into the sorted list. We cannot do
        this in the original list because then future structures may map to
        a newly inserted structure as opposed to the exterior polygon
    """
    offset = 0
    for item in previous:
        idx, n_pts = item
        if index > idx:
            offset += n_pts
    return offset


def split_by_holes(poly: np.ndarray) -> tuple:
    """
    Function
    ----------
    Given a numpy array boolean mask, the holes and polygon are split

    Parameters
    ----------
    poly : numpy.ndarray
        A 2D boolean mask array corresponding to the segmentation mask

    Returns
    ----------
    (numpy.ndarray, numpy.ndarray) if polygon has holes
    (None, numpy.ndarray) if polygon does not have holes
    """
    filled = scipy.ndimage.binary_fill_holes(poly)
    inner = filled ^ poly
    if np.sum(inner):
        return (inner, filled)
    return None, poly


def all_points_merged(poly: np.ndarray, merged: np.ndarray) -> bool:
    """
    Function
    ----------
    Determines if all points have been sorted and merged

    Parameters
    ----------
    poly : numpy.ndarray
        A 2D boolean mask array corresponding to the segmentation mask
    merged : numpy.ndarray
        An array of coordinates, in dimensions Nx3

    Returns
    ----------
    bool
        True if the number of coordinates in merged >= poly
        False if the number of coordinates in merged < poly

    Notes
    ----------
    Because we append to merged for each inner, when larger we have iterated
        through all the points, unless the holes constitute a single pixel,
        in which they are likely undesired anyways
    """
    mask = wire_mask(poly)
    total_pts = np.rollaxis(np.transpose(mask.nonzero()), 0, 2)
    return total_pts.shape[0] <= merged.shape[0]


def poly_to_coords_2D(poly: np.ndarray, ctcoord: np.ndarray,
                      flatten: bool = True,
                      invert: bool = False) -> np.ndarray:
    """
    Function
    ----------
    Given a 2D boolean array, returns an array of surface coordinates
        in the patient coordinate system. This is functional with filled
        polygons, polygons with hole(s) and nested holes and polygons

    Parameters
    ----------
    poly : numpy.ndarray
        A 2D boolean mask array corresponding to the segmentation mask
    ctcoord : numpy.ndarray
        A numpy array of the ct image coordinate system returned from
        utils.prepare_coordinate_mapping()
    flatten : bool (Default = True)
        Specifies if the returned coordinates are flattend into DICOM
        RT standard, as a single dimensional array
    invert : bool (Default = False)
        Specifies if the wire_mask function is inverted which is necessary
        for holes or polygons within holes

    Returns
    ----------
    numpy.ndarray
        An array of binary mask surface coordinates in the patient
        coordinate space, flattened into DICOM RT format

    Notes
    ----------
    Invert is called as the negation of the previous value. This is
        done in the case where a nested hole contains a polygon and
        the edge detection must alternate between erosion and dilation
    """
    inner, outer = split_by_holes(poly)
    # inner=None if polygon has no holes
    if inner is not None:
        inner_pts = poly_to_coords_2D(
            inner, ctcoord, flatten=False, invert=not invert)
        outer_pts = poly_to_coords_2D(
            outer, ctcoord, flatten=False, invert=invert)
        merged = merge_sorted_points(inner_pts, outer_pts)
        if all_points_merged(poly, merged) and not invert:
            return merged.flatten()
        return merged

    # Check if there are multiple polygons
    polygons, n_polygons = skm.label(poly, connectivity=2, return_num=True)
    if n_polygons > 1:
        i_polys = [polygons == n for n in range(
            1, n_polygons + 1)]  # 0=background
        return [poly_to_coords_2D(x, ctcoord, flatten=False, invert=True) for x in i_polys]

    # Convert poly to coords and sort, if no holes or multi-polygons
    mask = wire_mask(poly, invert=invert)
    points = np.rollaxis(
        ctcoord[tuple(mask.nonzero())+np.index_exp[:]].T, 0, 2)
    points_sorted = kd_sort_nearest(points)
    if flatten:
        return points_sorted.flatten()
    return points_sorted


def separate_polygons(z_slice: np.ndarray, mim: bool = True) -> [np.ndarray]:
    """
    Function
    ----------
    Given a z-slice, return each polygon to convert to a ContourSequence

    Parameters
    ----------
    z-slice : numpy.ndarray
        A 2D boolean numpy array of the segmentation mask
    mim : bool (Default = True)
        Creates MIM style contours if True
            MIM connects holes with a line of width zero
        Creates Pinnacle style contours if False
            Pinnacle creates holes as a separate structure

    Returns
    ----------
    [numpy.ndarray]
        A list of 2D boolean numpy arrays corresponding to each
        polygon to be converted into a Contour Sequence
    """
    each_polygon = []
    polygons, n_polygons = skm.label(z_slice, connectivity=2, return_num=True)
    for poly_index in range(1, n_polygons + 1):
        polygon = polygons == poly_index
        if mim:  # Mim doesn't separate holes
            each_polygon.extend([polygon])
        else:  # Pinnacle separates holes
            holes, filled = split_by_holes(polygon)
            if holes is not None:
                each_polygon.extend([filled])
                each_polygon.extend(separate_polygons(holes, mim=False))
            else:
                each_polygon.extend([polygon])
    return each_polygon


def nearest(array: np.ndarray, value: float) -> int:
    """
    Function
    ----------
    Finds the nearest index of array to a given value

    Parameters
    ----------
    array : np.array
    value : float

    Returns
    ----------
    index : int
    """
    return np.abs(np.asarray(array) - value).argmin()


def key_list_creator(key_list: list, *args) -> object:
    """
    Function
    ----------
    Smaller wrapper to create a key ordering list for the rtstruct func

    Parameters
    ----------
    key_list : list
        A list of the desired index order of structures, if present
    *args : list of arguments
        Arguments to be passed to key_list.index(*)

    Returns
    ----------
    Returns a function which yields the proper index of each structure
    """
    if isinstance(key_list, dict):
        key_list = list(key_list.keys())

    def new_key(*args):
        return key_list.index(*args)

    return new_key


def generate_instance_uid() -> str:
    """
    Function
    ----------
    Creates a randomly generated, MIM formatted instance UID string

    Parameters
    ----------
    None

    Returns
    ----------
    instance_uid : string
        An instance UID string which fits the format for MIM generated RTSTRUCTS

    Notes
    ----------
    The prefix of '2.16.840.1.' is MIM standard, the remainder is a
        randomly generated hash value

    References
    ----------
    Based on pydicom.uid.generate_uid() and documentation from MIM Software Inc.
    https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.uid.generate_uid.html
    """
    entropy_srcs = [
        str(uuid.uuid1()),  # 128-bit from MAC/time/randomness
        str(os.getpid()),  # Current process ID
        hex(random.getrandbits(64))  # 64 bits randomness
    ]

    # Create UTF-8 hash value
    hash_val = hashlib.sha512(''.join(entropy_srcs).encode('utf-8'))
    hv = str(int(hash_val.hexdigest(), 16))

    # Format all the SOP Instance UID stops properly
    terms = hv[:6], hv[6], hv[7:15], hv[15:26], hv[26:35], hv[35:38], hv[38:41]
    return '2.16.840.1.' + '.'.join(terms)
