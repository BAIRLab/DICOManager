#! /usr/bin/python
#
# This script will use a trained Beaunet model to run on a directory of DICOM-format image files (CT only) and
# write a DICOM RTSTRUCT file that contains predicted contours.
#

# May want to do less specific imports ... 
import concurrent.futures
import numpy as np
import pydicom
import scipy
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import repeat
from scipy.ndimage.morphology import binary_erosion
# TODO: Remove these subfucntion imports
#from pydicom import dataset, read_file, sequence, uid
# This is a strange import, clean up
#import argparse, h5py, numpy as np, os, random, time
#from skimage.transform import resize
#from skimage.measure import label, regionprops


"""
NOTE: Reconstruction plan:
use __prepare_coordinate_mapping to get image to patient coordinates
use __compute_surface2D to get the surface of a contour
use the interesction of those two to get the patient coordinates
    surface points
with those points, unravel them and project them into the DICOM tree
    with the correct additional information (UID Tags, Names, Colors)
when we generate the RTSTRUCT, we need a list of all image series, and
    their corresponding image slice locations, so we can save the UID
    for each slice in the DICOM header info
"""


def _prepare_coordinate_mapping(ct_hdr):
    """
    Function
    ----------
    Given a DICOM CT image slice, returns an array of pixel coordinates

    Parameters
    ----------
    ct_hdr : pydicom.dataset.FileDataset
        A CT dicom object to compute the image coordinate locations upon
        where _hdr means that the PixelData field is not required

    Returns
    ----------
    numpy.ndarray
        A numpy array of shape Mx2 where M is the dcm.Rows x dcm.Cols,
        the number of (x, y) pairs represnting coordinates of each pixel

    Notes
    ----------
    Computes M via DICOM Standard Equation C.7.6.2.1-1
        https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037
    Due to DICOM header orientation:
        D_i, D_j = (Column, Row)
        PixelSpacing = (Row, Column)
    """
    # Unpacking arrays is poor form, but I'm feeling rebellious...
    X_x, X_y, X_z = np.array(ct_hdr.ImageOrientationPatient[:3]).T
    Y_x, Y_y, Y_z = np.array(ct_hdr.ImageOrientationPatient[3:]).T
    S_x, S_y, S_z = np.array(ct_hdr.ImagePositionPatient)
    D_j, D_i = np.array(ct_hdr.PixelSpacing)
    j, i = np.indices((ct_hdr.Rows, ct_hdr.Columns))

    M = np.array([[X_x*D_i, Y_x*D_j, 0, S_x],
                  [X_y*D_i, Y_y*D_j, 0, S_y],
                  [X_z*D_i, Y_z*D_j, 0, S_z],
                  [0, 0, 0, 1]])

    C = np.array([i, j, 0, 1])

    # Returns coordinates in [x, y, 3], with [:-1] to reduce runtime
    return np.rollaxis(np.stack(np.dot(M[:-1], C)), 0, 3)


def _wire_mask(arr):
    """
    Function
    ----------
    Given an 2D boolean array, returns those pixels on the surface

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D array corresponding to the segmentation mask

    Returns
    ----------
    numpy.ndarray
        A 2D boolean array, with points representing the contour surface

    Notes
    ----------
    The resultant surface is not the minimium points to fully enclose the
        surface, like the standard DICOM RTSTRUCT uses. Instead this gives
        all points along the surface. Should construct the same, but this
        calculation is faster than using Shapely. 
    Could also use skimage.measure.approximate_polygon(arr, 0.001)
        To compute an approximate polygon. Better yet, do it after doing
        the binary erosion techniqe, and compare that to the speed of 
        using shapely to find the surface
    The most accurate way would be with ITK using: 
        itk.GetImageFromArray
        itk.ContourExtractor2DImageFilter
    """
    if arr.dtype != 'bool':
        arr = np.array(arr, dtype='bool')
    return binary_erosion(arr) ^ arr


def _array_to_coords_2D(arr, ct_hdr, flatten=True):
    """
    Function
    ----------
    Given 2D boolean array, eturns an array of surface coordinates in
        the patient coordinate system

    Parameters
    ----------
    ct_hdr : pydicom.dataset.FileDataset
        A CT dicom object to compute the image coordinate locations upon
        where _hdr means that the PixelData field is not required
    arr : numpy.ndarray
        A 2D boolean mask array corresponding to the segmentation mask
    flatten : bool (Default = True)
        Specifies if the returned coordinates are flattened into DICOM
        RT standard, as single dimensional array

    Returns
    ----------
    numpy.ndarray
        An array of binary mask surface coordinates in the patient
        coordinate space. Flattened, if specified, into DICOM RT format
    
    Raises
    ----------
    AssertionError
        Raised if arr.ndim is not 2
        Raised if ct_dcm is not a pydicom dataset 
    """
    # These will be helpful for my troubleshooting later in the process
    # after which, they will likely become unnecessary due to protections upstream
    assert arr.ndim != 2, 'The input boolean mask is not 2D'
    assert type(ct_hdr) is pydicom.dataset.FileDataset, 'ct_dcm is not a pydicom dataset'

    mask = _wire_mask(arr)
    coords = _prepare_coordinate_mapping(ct_hdr)

    if not flatten:
        return coords[tuple(mask.nonzero()) + np.index_exp[:]].T
    return coords[tuple(mask.nonzero()) + np.index_exp[:]].T.flatten()


def _slice_thickness(dcm0, dcm1):
    """
    Function
    ----------
    Computes the slice thickness for a DICOM set
        NOTE Calculates based on slice location and instance number
        Does not trust SliceThickness DICOM Header

    Parameters
    ----------
    dcm0, dcm1 : str or pydicom.dataset.FileDataset
        Either a string to the dicom path or a pydicom dataset

    Returns
    ----------
    slice_thickness : float
        A float representing the robustly calculated slice thickness
    """

    if type(dcm0) != pydicom.dataset.FileDataset:
        dcm0 = pydicom.dcmread(dcm0)
    if type(dcm1) != pydicom.dataset.FileDataset:
        dcm1 = pydicom.dcmread(dcm1)

    loc0 = dcm0.SliceLocation
    loc1 = dcm1.SliceLocation
    inst0 = dcm0.InstanceNumber
    inst1 = dcm1.InstanceNumber

    return abs((loc1-loc0) / (inst1-inst0))


# This is coming from reconstruction
# TODO: #8 Move common utilities into a shared library for re and deconstruction
def _img_dims(dicom_list):
    """
    Function
    ----------
    Computation of the image dimensions for slice thickness and number
        of z slices in total

    Parameters
    ----------
    dicom_list : list
        A list of the paths to every dicom for the given image

    Returns
    ----------
    (thickness, n_slices, low, high, flip) : (float, int, float, float, boolean)
        0.Slice thickness computed from dicom locations, not header
        1.Number of slices, computed from dicom locations, not header
        2.Patient coordinate system location of lowest instance
        3.Patient coordinate system location of highest instance
        4.Boolean indicating if image location / instances are flipped

    Notes
    ----------
    The values of high and low are for the highest and lowest instance,
        meaning high > low is not always true
    """
    # We need to save the location and instance, to know if counting up or down
    low = [float('inf'), 0]
    high = [-float('inf'), 0]

    instances = []
    for f in dicom_list:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if float(ds.SliceLocation) < low[0]:
            low = [float(ds.SliceLocation), round(ds.InstanceNumber)]
        if float(ds.SliceLocation) > high[0]:
            high = [float(ds.SliceLocation), round(ds.InstanceNumber)]
        instances.append(round(ds.InstanceNumber))
    instances.sort()

    thickness = _slice_thickness(dicom_list[0], dicom_list[1])
    n_slices = 1 + (abs(high[0]-low[0]) / thickness)

    # We need to cover for if instance 1 is missing
    # Unfortunately, we don't know how many upper instances could be missing
    if 1 != min(instances):
        diff = min(instances) - 1
        n_slices += diff
        if low[1] < high[1]:
            low[0] -= thickness * diff
        else:
            high[0] += thickness * diff

    flip = True
    if low[1] > high[1]:
        flip = False
        low[0], high[0] = high[0], low[0]

    return thickness, round(n_slices), low[0], high[0], flip


def _get_z_loc(ct_hdr, ct_thick, ct_loc0):
    return round(abs((ct_loc0-ct_hdr.SliceLocation) / ct_thick))


# A function to collect all the UIDs and store them as a dictionary with keyword
# as the image coordinate z-slice location
def _generate_uid_dict(ct_series):
    uid_dict = {}
    #ct_thick, _, ct_loc0, _, _ = _img_dims(ct_series)
    for ct in ct_series:
        hdr = pydicom.dcmread(ct, stop_before_pixels=True)
        ct_store = pydicom.dataset.Dataset()
        ct_store.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')
        ct_store.ReferencedSOPInstanceUID = hdr.SOPInstanceUID
        uid_dict.update({hdr.SOPInstanceUID: ct_store})
        #uid_dict.update({_get_z_loc(hdr, ct_thick, ct_loc0): hdr.InstanceUID})
        #May need to save the z-axis location as well...
    return dict(sorted(uid_dict.items()))


# I don't know if this will work...
def _initialize_rt_dcm(ct_series):
    # TODO: Order the initializtion of header in the order of header fields
    # Start crafting the RTSTRUCT
    rt_dcm = pydicom.dataset.Dataset()
    # Read the ct to build the header from
    ct_dcm = pydicom.dcmread(ct_series[0], stop_before_pixels=True)

    # Time specific DICOM header info
    current_time = time.localtime()
    rt_dcm.SpecificCharacterSet = 'ISO_IR 100'
    # DICOM date format
    rt_dcm.InstanceCreationDate = time.strftime('%Y%m%d', current_time)
    # DICOM time format
    rt_dcm.InstanceCreationTime = time.strftime('%H%M%S.%f', current_time)[:-3]
    rt_dcm.StructureSetDate = time.strftime('%Y%m%d', current_time)
    rt_dcm.StructureSetTime = time.strftime('%H%M%S.%f', current_time)[:-3]

    # UID Header Info
    # RT Structure Set Storage Class
    rt_dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    rt_dcm.SOPInstanceUID = pydicom.uid.generate_uid()
    rt_dcm.Modality = 'RTSTRUCT'
    rt_dcm.Manufacturer = 'Beaumont Health'
    rt_dcm.ManufacturersModelName = 'Beaunet Artificial Intelligence Lab'
    rt_dcm.StructureSetLabel = 'Auto-Segmented Contours'
    rt_dcm.StructureSetName = 'Auto-Segmented Contours'
    rt_dcm.SoftwareVersions = ['0.1.0']

    # Referenced study
    ref_study_ds = pydicom.dataset.Dataset()
    # Study Component Management SOP Class
    ref_study_ds.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.3.1.2.3.2')
    ref_study_ds.ReferencedSOPInstanceUID = ct_dcm.StudyInstanceUID
    rt_dcm.ReferencedStudySequence = pydicom.sequence.Sequence([ref_study_ds])

    # Demographics
    if 'PatientsName' in ct_dcm:
        rt_dcm.PatientsName = ct_dcm.PatientsName
    else:
        rt_dcm.PatientsName = 'UNKNOWN^UNKNOWN^^'

    if 'PatientID' in ct_dcm:
        rt_dcm.PatientID = ct_dcm.PatientID
    else:
        rt_dcm.PatientID = '0000000'

    if 'PatientsBirthDate' in ct_dcm:
        rt_dcm.PatientsBirthDate = ct_dcm.PatientsBirthDate
    else:
        rt_dcm.PatientsBirthDate = ''

    if 'PatientsSex' in ct_dcm:
        rt_dcm.PatientsSex = ct_dcm.PatientsSex
    else:
        rt_dcm.PatientsSex = ''

    # This study
    rt_dcm.StudyInstanceUID = ct_dcm.StudyInstanceUID
    rt_dcm.SeriesInstanceUID = pydicom.uid.generate_uid()

    if 'StudyID' in ct_dcm:
        rt_dcm.StudyID = ct_dcm.StudyID
    if 'SeriesNumber' in ct_dcm:
        rt_dcm.SeriesNumber = ct_dcm.SeriesNumber
    if 'StudyDate' in ct_dcm:
        rt_dcm.StudyDate = ct_dcm.StudyDate
    if 'StudyTime' in ct_dcm:
        rt_dcm.StudyTime = ct_dcm.StudyTime

    # Referenced frame of reference
    ref_frame_of_ref_ds = pydicom.dataset.Dataset()
    rt_dcm.ReferencedFrameOfReferenceSequence = pydicom.sequence.Sequence([ref_frame_of_ref_ds])
    ref_frame_of_ref_ds.FrameOfReferenceUID = ct_dcm.FrameOfReferenceUID

    # Referenced study
    # TODO: Determine difference between this and the above call
    rt_ref_study_ds = pydicom.dataset.Dataset()
    ref_frame_of_ref_ds.RTReferencedStudySequence = pydicom.sequence.Sequence([rt_ref_study_ds])
    # Study Component Management SOP Class
    rt_ref_study_ds.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.3.1.2.3.2')
    rt_ref_study_ds.ReferencedSOPInstanceUID = ct_dcm.StudyInstanceUID

    # Referenced sereis
    rt_ref_series_ds = pydicom.dataset.Dataset()
    rt_ref_study_ds.RTReferencedSeriesSequence = pydicom.sequence.Sequence([rt_ref_series_ds])
    rt_ref_series_ds.SeriesInstanceUID = ct_dcm.SeriesInstanceUID
    ct_uids = _generate_uid_dict(ct_series) 
    rt_ref_series_ds.ContourImageSequence = pydicom.sequence.Sequence([list(ct_uids.values())])
    ''' 
    # More explicit way of saving Image UIDs
    rt_ref_series_ds.ContourImageSequence = pydicom.sequence.Sequence()

    ct_uids = _generate_uid_dict(ct_series) 
    for uid in ct_uids:
        rt_ref_series_ds.ContourImageSequence.append(ct_uids[uid])
    '''
    # TODO: Return all the ref frame, study and series, or find a way to integrate them into the dcm

    rt_dcm.StructureSetROISequence = pydicom.sequence.Sequence()
    rt_dcm.ROIContourSequence = pydicom.sequence.Sequence()
    rt_dcm.RTROIObservationsSequence = pydicom.sequence.Sequence()
    
    file_meta = pydicom.dataset.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    # Need to investigate what this does. 
    output_ds = pydicom.dataset.FileDataset(None, rt_dcm, file_meta=file_meta, preamble="\0" * 128)
    return output_ds 

# A function which takes the RT Struct and wire mask and appends to the rt file
# This should have a uid_list which is a list of the uid values per contour list  
def _append_contour_to_dcm(source_rt, coords_list, uid_list, roi_name):
    roi_number = len(source_rt.StructureSetROISequence) + 1
    # Add ROI to Structure Set Sequence
    str_set_roi = pydicom.dataset.Dataset()
    str_set_roi.ROINumber = roi_number
    str_set_roi.ROIName = roi_name
    str_set_roi.ReferencedFrameOfReferenceUID = source_rt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
    str_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
    source_rt.StructureSetROISequence.append(str_set_roi)

    # Add ROI to ROI Contour Sequence
    roi_contour_ds = pydicom.dataset.Dataset()
    roi_contour_ds.ROIDisplayColor = [255, 0, 0] 
    roi_contour_ds.ReferencedROINumber = roi_number
    roi_contour_ds.ContourSequence = pydicom.sequence.Sequence([])
    source_rt.ROIContourSequence.append(roi_contour_ds)
    
    for index, coords in enumerate(coords_list):
        contour_ds = pydicom.dataset.Dataset()
        # TODO: This is where we need the UIDs from the slices.
        contour_ds.ContourImageSequence = pydicom.sequence.Sequence(uid_list[index])
        contour_ds.ContourGeometricType = 'CLOSED_PLANAR'
        contour_ds.NumberOfContourPoints = len(coords) // 3
        contour_ds.ContourData = [f'{p:0.2f}' for p in coords]
        roi_contour_ds.ContourSequence.append(contour_ds)

    # Add ROI to RT ROI Observations Sequence
    rt_roi_obs = pydicom.dataset.Dataset()
    rt_roi_obs.ObservationNumber = roi_number
    rt_roi_obs.ReferencedROINumber = roi_number
    rt_roi_obs.ROIObservationLabel = roi_name
    rt_roi_obs.RTROIInterpretedType = 'ORGAN'
    source_rt.RTROIObservationsSequence.append(rt_roi_obs)
    
    return source_rt


# A function to add to an RT
def to_rt(source_rt, ct_series, contour_array, roi_name_list=None):
    ct_hdr = pydicom.dcmread(ct_series[0], stop_before_pixels=True)
    # Need to have the contour slices in relation to the individual image UIDs
    # We can do that with the help of the z-axis location data. Will likely 
    # want to have the data stored in a nice-to-use struct
    # Could also find the nearest image slice to a given contour location
    # I think I will try this until I know that the robustness of this solution
    for index, contour in enumerate(contour_array):
        if roi_name_list:
            roi_name = roi_name_list[index]
        else:
            roi_name = 'GeneratedContour' + str(index)
        coords = _array_to_coords_2D(contour, ct_hdr)
        # We could probably save this from being called twice...
        uid_dict = _generate_uid_dict(ct_series)
        source_rt = _append_contour_to_dcm(source_rt, coords, uid_dict, roi_name)  
    return False

# A function to create new from an RT
def from_rt(source_rt, ct_series, contour_array, roi_name_list=None):
    return False

# Could make this into an actual wrapper, instead of what it is now...
def from_ct(ct_series, contour_array, roi_name_list=None):
    # Essentially this is the same as to_rt except we need to build a fresh RT
    rt_dcm = _initialize_rt_dcm(ct_series)
    rt_dcm = to_rt(rt_dcm, ct_series, contour_array, roi_name_list)
    return rt_dcm
