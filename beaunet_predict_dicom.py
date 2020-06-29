#! /usr/bin/python
#
# This script will use a trained Beaunet model to run on a directory of DICOM-format image files (CT only) and
# write a DICOM RTSTRUCT file that contains predicted contours.
#
'''
TODO:
Finish with the construction of the rtstruct header
Determine how Leonid used shapely to compute the surface points. (poly.points.exterior, it looks like)
Assert that the Polygon reconstruction is equaivalent between Leonid's and my method
Determine how Leonid packed the points into the RTSTRUCT file format
'''

# May want to do less specific imports ... 
from scipy.ndimage.morphology import binary_erosion
import scipy
import numpy as np
# This is a strange import, clean up
import argparse, h5py, numpy as np, os, random, time
from functools import partial
from itertools import repeat
import concurrent.futures
import pydicom
from dataclasses import dataclass
# TODO: Remove these subfucntion imports
#from pydicom import dataset, read_file, sequence, uid

from skimage.transform import resize
from skimage.measure import label, regionprops

from datetime import datetime

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

    return rt_dcm 

# A function which takes the RT Struct and wire mask and appends to the rt file
def _append_contour_to_dcm(source_rt, coords):

    return False


# A function to add to an RT
def to_rt(source_rt, ct_series, contour_array):
    ct_hdr = pydicom.dcmread(ct_series[0], stop_before_pixels=True)
    for contour in contour_array:
        coords = _array_to_coords_2D(contour, ct_hdr)
        source_rt = _append_contour_to_dcm(source_rt, coords)  
    return False

# A function to create new from an RT
def from_rt(source_rt, ct_series, contour_array):
    return False

def from_ct(ct_series, contour_array):
    # Essentially this is the same as to_rt except we need to build a fresh RT
    rt_dcm = _initialize_rt_dcm(ct_series)
    rt_dcm = to_rt(rt_dcm, ct_series, contour_array)
    return rt_dcm


# A function to create new from CT
def from_ct(ct_series):
    # Start crafting the RTSTRUCT
    rt_dcm = pydicom.dataset.Dataset()

    # Time specific DICOM header info
    current_time = time.localtime()
    rt_dcm.SpecificCharacterSet = 'ISO_IR 100'
    rt_dcm.InstanceCreationDate = time.strftime('%Y%m%d', current_time) # DICOM date format
    rt_dcm.InstanceCreationTime = time.strftime('%H%M%S.%f', current_time)[:-3] # DICOM time format
    rt_dcm.StructureSetDate = time.strftime('%Y%m%d', current_time)
    rt_dcm.StructureSetTime = time.strftime('%H%M%S.%f', current_time)[:-3]

    # UID Header Info
    rt_dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # RT Structure Set Storage Class
    rt_dcm.SOPInstanceUID = uid.generate_uid()
    rt_dcm.Modality = 'RTSTRUCT'
    rt_dcm.Manufacturer = 'Beaumont Health'
    rt_dcm.ManufacturersModelName = 'Beaunet Artificial Intelligence Lab'
    rt_dcm.StructureSetLabel = 'Auto-Segmented Contours'
    rt_dcm.StructureSetName = 'Auto-Segmented Contours'
    rt_dcm.SoftwareVersions = ['0.1.0']

    # Scan through input directory and grab common data; first file will be used to initialize demographics
    # and study-level variables
    rtstruct_init = False
    contour_image_uids = {}
    roi_images = {}

    for image_file_name in filenames:
        # Don't try to force a read; instead, just attempt, and if it doesn't work,
        # or if this is not a CT slice, go to the next image.
        image_ds = read_file(os.path.join(dirpath, image_file_name), force=False)
        if not image_ds.SOPClassUID == uid.UID('1.2.840.10008.5.1.4.1.1.2'):
            continue
        if not rtstruct_init:
            # Referenced study
            referenced_study_ds = dataset.Dataset()
            referenced_study_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.3.1.2.3.2') # Study Component Management SOP Class
            referenced_study_ds.ReferencedSOPInstanceUID = image_ds.StudyInstanceUID
            rt_dcm.ReferencedStudySequence = sequence.Sequence([referenced_study_ds])
            # Demographics
            if 'PatientsName' in image_ds:
                rt_dcm.PatientsName = image_ds.PatientsName
            else:
                rt_dcm.PatientsName = 'UNKNOWN^UNKNOWN^^'
            if 'PatientID' in image_ds:
                rt_dcm.PatientID = image_ds.PatientID
            else:
                rt_dcm.PatientID = '0000000'

            if 'PatientsBirthDate' in image_ds:
                rt_dcm.PatientsBirthDate = image_ds.PatientsBirthDate
            else:
                rt_dcm.PatientsBirthDate = ''

            if 'PatientsSex' in image_ds:
                rt_dcm.PatientsSex = image_ds.PatientsSex
            else:
                rt_dcm.PatientsSex = ''

            # This study
            rt_dcm.StudyInstanceUID = image_ds.StudyInstanceUID
            rt_dcm.SeriesInstanceUID = uid.generate_uid()

            if 'StudyID' in image_ds:
                rt_dcm.StudyID = image_ds.StudyID
            if 'SeriesNumber' in image_ds:
                rt_dcm.SeriesNumber = image_ds.SeriesNumber
            if 'StudyDate' in image_ds:
                rt_dcm.StudyDate = image_ds.StudyDate
            if 'StudyTime' in image_ds:
                rt_dcm.StudyTime = image_ds.StudyTime

            # Referenced frame of reference
            referenced_frame_of_reference_ds = dataset.Dataset()
            rt_dcm.ReferencedFrameOfReferenceSequence = sequence.Sequence([referenced_frame_of_reference_ds])
            referenced_frame_of_reference_ds.FrameOfReferenceUID = image_ds.FrameOfReferenceUID
            rt_referenced_study_ds = dataset.Dataset()
            referenced_frame_of_reference_ds.RTReferencedStudySequence = sequence.Sequence([rt_referenced_study_ds])
            rt_referenced_study_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.3.1.2.3.2') # Study Component Management SOP Class
            rt_referenced_study_ds.ReferencedSOPInstanceUID = image_ds.StudyInstanceUID
            rt_referenced_series_ds = dataset.Dataset()
            rt_referenced_study_ds.RTReferencedSeriesSequence = sequence.Sequence([rt_referenced_series_ds])
            rt_referenced_series_ds.SeriesInstanceUID = image_ds.SeriesInstanceUID
            rt_referenced_series_ds.ContourImageSequence = sequence.Sequence()

            # Once all is done, mark initialized
            rtstruct_init = True

        # We only store the final contourimagesequence after all is done to maintain ordering
        contour_image_ds = dataset.Dataset()
        contour_image_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.5.1.4.1.1.2') # CT Image Storage
        contour_image_ds.ReferencedSOPInstanceUID = image_ds.SOPInstanceUID
        contour_image_uids[image_ds.SOPInstanceUID] = {'ds':contour_image_ds, 'filename':image_file_name, 'key': image_ds.SOPInstanceUID}

    # All images scanned (not processed!), build more sequences
    # This is all the images in the referenced series, regardless of whether they have contours or not
    rt_referenced_series_ds.ContourImageSequence.extend([contour_image_uids[key]['ds'] for key in sorted(contour_image_uids.keys())])

    # Create sequence objects for contours - these may be empty if no contours are created
    rt_dcm.StructureSetROISequence = sequence.Sequence()
    rt_dcm.ROIContourSequence = sequence.Sequence()
    rt_dcm.RTROIObservationsSequence = sequence.Sequence()
    # -------------------- Where _initialize_rt ends and other functions begin to be sourced ------------------------------------
    # Process the images one by one in order of contour sequence.
    image_num = 0
    image_batch = []
    image_processing_order = sorted(contour_image_uids.keys())

    for image_file_name, _ in [(contour_image_uids[key]['filename'], contour_image_uids[key]['key']) for key in image_processing_order]:
        try:
            image_ds = read_file(os.path.join(dirpath, image_file_name))
            # Do work if this DICOM file is a CT slice
            if image_ds.SOPClassUID == uid.UID('1.2.840.10008.5.1.4.1.1.2'):
                if len(image_batch) >= 24:
                    predict_slices_contours = _predict_slices(image_batch, model, mean_pixel, class_list)
                    # Add predictions to general list
                    for slice_id, contour_dict in predict_slices_contours.iteritems():
                        contour_image_uids[slice_id]['contours'] = {}
                        for roi_name, contour_list in contour_dict.iteritems():
                            contour_image_uids[slice_id]['contours'][roi_name] = contour_list
                            if roi_name not in roi_images:
                                roi_images[roi_name] = []
                            roi_images[roi_name].append([slice_id, contour_list])
                    image_num += len(image_batch)
                    print('    Processed slices {0:d} of {1:d}'.format(image_num, len(image_processing_order)))
                    image_batch = []
                image_batch.append(image_ds)
        except Exception as e:
            print('Exception: {0:s}'.format(e.message))

    predict_slices_contours = _predict_slices(image_batch, model, mean_pixel, class_list)
    # Add predictions to general list

    for slice_id, contour_dict in predict_slices_contours.iteritems():
        contour_image_uids[slice_id]['contours'] = {}
        for roi_name, contour_list in contour_dict.iteritems():
            contour_image_uids[slice_id]['contours'][roi_name] = contour_list
            if roi_name not in roi_images:
                roi_images[roi_name] = []
                roi_images[roi_name].append([slice_id, contour_list])
        image_num += len(image_batch)
        print('    Processed slices {0:d} of {1:d}'.format(image_num, len(image_processing_order)))

        # Now we have contours for each CT slice. We have a mapping of CT slice ID to data:
        #   contour_image_uids[ID] = {ds: image_ds, filename: str, key: ID, contours: {roi_name: list}}
        # And a mapping of ROI name to slices:
        #   roi_images[roi_name] = [[ID, contourlist], ...]
        # Let's construct the tags.

        rt_dcm.StructureSetROISequence = sequence.Sequence([])
        rt_dcm.ROIContourSequence = sequence.Sequence([])
        rt_dcm.RTROIObservationsSequence = sequence.Sequence([])
        roi_number = 1

        for roi_name in class_list:
            # If the ROI exists in our image set...
            if roi_name in roi_images:
                # Add ROI to Structure Set Sequence
                structure_set_roi_ds = dataset.Dataset()
                rt_dcm.StructureSetROISequence.append(structure_set_roi_ds)
                structure_set_roi_ds.ROINumber = roi_number
                structure_set_roi_ds.ReferencedFrameOfReferenceUID = rt_dcm.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                structure_set_roi_ds.ROIName = roi_name
                structure_set_roi_ds.ROIGenerationAlgorithm = 'AUTOMATIC'

                # Add ROI to ROI Contour Sequence
                roi_contour_ds = dataset.Dataset()
                rt_dcm.ROIContourSequence.append(roi_contour_ds)
                roi_contour_ds.ROIDisplayColor = list(color_dict[roi_name][1:])
                roi_contour_ds.ReferencedROINumber = roi_number
                roi_contour_ds.ContourSequence = sequence.Sequence([])

                for slice_id, contour_list in roi_images[roi_name]:
                    for contour in contour_list:
                        contour_ds = dataset.Dataset()
                        contour_ds.ContourImageSequence = sequence.Sequence([contour_image_uids[slice_id]['ds']])
                        contour_ds.ContourGeometricType = 'CLOSED_PLANAR'
                        contour_ds.NumberOfContourPoints = len(contour)
                        contour_ds.ContourData = ['{0:0.2f}'.format(val) for p in contour for val in p]
                        roi_contour_ds.ContourSequence.append(contour_ds)

                # Add ROI to RT ROI Observations Sequence
                rt_roi_obs = dataset.Dataset()
                rt_dcm.RTROIObservationsSequence.append(rt_roi_obs)
                rt_roi_obs.ObservationNumber = roi_number
                rt_roi_obs.ReferencedROINumber = roi_number
                rt_roi_obs.ROIObservationLabel = roi_name
                rt_roi_obs.RTROIInterpretedType = 'ORGAN'

                # Update ROI number
                roi_number += 1

        file_meta = dataset.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = uid.generate_uid()
        file_meta.ImplementationClassUID = uid.generate_uid()

        output_ds = dataset.FileDataset(output_path, {}, file_meta=file_meta, preamble="\0" * 128)
        output_ds.update(rt_dcm)
        output_ds.save_as(output_path)

    return True
