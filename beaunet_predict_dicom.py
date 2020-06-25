#! /usr/bin/python
#
# This script will use a trained Beaunet model to run on a directory of DICOM-format image files (CT only) and
# write a DICOM RTSTRUCT file that contains predicted contours.
#


# May want to do less specific imports ... 
from scipy.ndimage.morphology import binary_erosion
import scipy
import numpy as np
import argparse, h5py, numpy as np, os, random, time
from functools import partial
from itertools import repeat
import concurrent.futures
import pydicom
# TODO: Remove these subfucntion imports
from pydicom import dataset, read_file, sequence, uid

# TODO : Identify how Shapely is used, it is quite slow...
from shapely import speedups
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

from skimage.transform import resize
from skimage.measure import label, regionprops

from datetime import datetime

# Make shapely go fast
if speedups.available:
    speedups.enable()

# For randomness
random.seed(datetime.now())
np.random.seed(datetime.now())

"""
NOTE: Reconstruction plan:
use __prepare_coordinate_mapping to get image to patient coordinates
use __compute_surface2D to get the surface of a contour
use the interesction of those two to get the patient coordinates
    surface points
with those points, unravel them and project them into the DICOM tree
    with the correct additional information (UID Tags, Names, Colors)

NOTE: IMPORTED ACCESSORY FILES AND DIRS
color_dictionary.txt : declares a mapping from seg type to RGB color value
pixel_stats.txt : Contains "mean_pixel	203.629169"
anonymization.txt : Contains a mapping from patient to encoded number
dataset.hdf5 : A h5 file of the constructed dataset
Classifications/ : A directory of the segmentation classifications of each
                   encoded, anonymized patient file
DICOMImages/ : A directory of all anonymized encoded dicom images
OverlayImages/ : A directory of all encoded 'overlay images' in .png
Segmentatoins/ : A directory of segmentations saved a .png files
                 This indicates he may not have been able to save RTSTRUCT .dcms

CHANGES: 
1. Remove color dict and make all imported contours Red [255, 0, 0] 
2. Remove pixel_stats.txt import

Will nest these all within the deconstruction.py file / class
"""


def _prepare_coordinate_mapping(ct_dcm):
    """
    Function
    ----------
    Given a DICOM CT image slice, returns an array of pixel coordinates

    Parameters
    ----------
    ct_dcm : pydicom.dataset.FileDataset
        A CT dicom object to compute the image coordinate locations upon

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
    X_x, X_y, X_z = np.array(ct_dcm.ImageOrientationPatient[:3]).T
    Y_x, Y_y, Y_z = np.array(ct_dcm.ImageOrientationPatient[3:]).T
    S_x, S_y, S_z = np.array(ct_dcm.ImagePositionPatient)
    D_j, D_i = np.array(ct_dcm.PixelSpacing)
    j, i = np.indices((ct_dcm.Rows, ct_dcm.Columns))

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
    """
    if arr.dtype != 'bool':
        arr = np.array(arr, dtype='bool')
    return binary_erosion(arr) ^ arr


def _array_to_coords_2D(arr, ct_dcm, flatten=True):
    """
    Function
    ----------
    Given 2D boolean array, eturns an array of surface coordinates in
        the patient coordinate system

    Parameters
    ----------
    ct_dcm : pydicom.dataset.FileDataset
        A CT dicom object to compute the image coordinate locations upon
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
    # TODO: Check if this should be dataset of fileset ... 
    assert type(ct_dcm) is pydicom.dataset.Dataset, 'ct_dcm is not a pydicom dataset'

    mask = _wire_mask(arr)
    coords = _prepare_coordinate_mapping(ct_dcm)

    if not flatten:
        return coords[tuple(mask.nonzero()) + np.index_exp[:]].T
    return coords[tuple(mask.nonzero()) + np.index_exp[:]].T.flatten()


def _threaded_geom_op(item):
    try:
        image_ds = item[0]
        y_elem = item[1]
        class_list = item[2]

        y_elem = resize(y_elem, output_shape=(image_ds.Rows, image_ds.Columns), order=0, mode='constant', preserve_range=True)

        # Process prediction into contours
        contour_list_pred = {}  # This will contain contours for this slice represented as lists of polygons
        p_s_x = float(image_ds.PixelSpacing[0])
        p_s_y = float(image_ds.PixelSpacing[1])
        p_coords = _prepare_coordinate_mapping(
            image_ds)  # X,Y,3 matrix containing at each voxel the coordinates of the center of that voxel

        for roi_name, label_img in zip(class_list[1:], np.rollaxis(y_elem, 2, 0)[1:, ...]):
            connected_comps_pred = label(label_img, connectivity=2)
            contour_list_roi = []
            region_count = 0
            
            for region in regionprops(connected_comps_pred):
                region_count += 1
                multi_poly_list = []

                for point in MultiPoint([(p_coords[p[0], p[1], :]) for p in region.coords]):
                    # We have converted now from pixel coordinates to RCS coordinates. When
                    # we divide pixelspacing by 2, we get all sorts of ugly disjoint contours.
                    # Instead, we divide by 1.5 for a bit of overlap.
                    point_poly = Polygon([(point.x - p_s_x / 1.5, point.y - p_s_y / 1.5, point.z),
                                          (point.x + p_s_x / 1.5, point.y - p_s_y / 1.5, point.z),
                                          (point.x + p_s_x / 1.5, point.y + p_s_y / 1.5, point.z),
                                          (point.x - p_s_x / 1.5, point.y + p_s_y / 1.5, point.z)])

                    multi_poly_list.append(point_poly)

                dissolve_poly = unary_union(MultiPolygon(multi_poly_list))
                shape_polys = []

                if isinstance(dissolve_poly, Polygon):
                    shape_polys.append(dissolve_poly)
                elif isinstance(dissolve_poly, MultiPolygon):
                    shape_polys.extend([poly for poly in dissolve_poly])

                # Old and busted: This region's shapes get coordified at the time of region processing.
                # for poly in shape_polys:
                #     coord_list = poly.exterior.coords[:]
                #     coord_list = [(x[0], x[1], x[2]) for x in coord_list]
                #     contour_list_roi.append(coord_list)

                # New hotness: We store the polygons for the region and postprocess the whole ROI at once.

                contour_list_roi.extend(shape_polys)

            # If there are any contours for this ROI, we'll add it to the list of contours for this slice
            # Postprocessing based on contour geometry, etc, may go here if we store it somewhere.

            if contour_list_roi:
                postprocessed_roi_list = contour_list_roi

                if POSTPROCESS:
                    # Step 1: Eliminate free pixels.
                    #    A 'free' pixel is a polygon with area less than min_poly_area, and
                    #    more than max_poly_dist away from its closest neighbor.

                    min_poly_area = 5.5 # Hardcoded magic number, will change with resolution/upsampling
                    max_poly_dist = max(p_s_x, p_s_y)

                    # We find all polys smaller than our cutoff, and measure the minimum distance between them and any
                    # other poly in this ROI (that is not them)

                    small_polys = [(poly, min([poly.distance(cand_poly) for cand_poly in contour_list_roi])) for poly in contour_list_roi if poly.area < min_poly_area]
                    big_polys = [poly for poly in contour_list_roi if poly.area >= min_poly_area]

                    for poly, min_dist in small_polys:
                        if min_dist > 0.0 and min_dist < max_poly_dist:
                            big_polys.append(poly) # A small poly that is closer than max_poly_dist isn't really small.

                    postprocessed_roi_list = big_polys

                    if not postprocessed_roi_list:
                        continue # Don't add anything if there aren't contours left

                    # Step 2: Close gaps. We dilate and erode polygons using buffers.

                    buffer_radius = 5*max_poly_dist
                    current_z = postprocessed_roi_list[0].exterior.coords[0][2]
                    buffer_poly = MultiPolygon(postprocessed_roi_list).buffer(buffer_radius).buffer(-buffer_radius)

                    if isinstance(buffer_poly, MultiPolygon):
                        postprocessed_roi_list = [Polygon([(p[0], p[1], current_z) for p in poly.exterior.coords[:]]) for poly in buffer_poly]
                    elif isinstance(buffer_poly, Polygon):
                        postprocessed_roi_list = [Polygon([(p[0], p[1], current_z) for p in buffer_poly.exterior.coords[:]])]
                    if not postprocessed_roi_list:
                        continue # Don't add anything if there aren't contours left

                    # Step 3: ROI-specific contour number restriction
                    if roi_name == 'RtLung' or roi_name == 'LtLung':
                        # Up to 3 lung contours allowed - take biggest by area
                        if len(postprocessed_roi_list) > 3:
                            postprocessed_roi_list = sorted(postprocessed_roi_list, key=lambda x: x.area, reverse=True)[0:3]
                    else:
                        # Up to 1 contour allowed for most structures - take biggest by area
                        if len(postprocessed_roi_list) > 1:
                            postprocessed_roi_list = [sorted(postprocessed_roi_list, key=lambda x: x.area, reverse=True)[0]]

                # Set list of geometries for this ROI
                contour_list_pred[roi_name] = postprocessed_roi_list

        #Additional postprocessing based on ROI-on-ROI interactions goes here
        output_dict = {}
        for k,v in contour_list_pred.iteritems():
            output_dict[k] = [poly.exterior.coords[:] for poly in v]

        return output_dict

    except:
        print('Error in threaded op')

# Can likely remove most, but will check first
def _predict_slices(images, model, mean_pixel, class_list):
    """
    Core prediction function. Uses the Keras model in GPU mode to generate probability maps for each slice, then
    uses Shapely geometry processing with multiprocessing to convert those raster maps to vector data.
    :param images: A list of pydicom Datasets representing the CT slices to predict contours for
    :param model: A pre-trained and initialized Beaunet Keras model
    :param class_list: A list of contour names in order describing the output vector of the Keras model
    :return: A dictionary mapping slice UIDs to ROI names to a list of contours for that ROI.
    """

    # Process images in sequence
    slices_to_contours = {}

    executor = concurrent.futures.ProcessPoolExecutor(24)
    futures = [executor.submit(_threaded_geom_op, item) for item in zip(images, y_data, repeat(class_list, len(images)))]
    concurrent.futures.wait(futures)
    results = [output.result() for output in futures]

    for image_ds, predicted_contours in zip(images, results):
        slices_to_contours[image_ds.SOPInstanceUID] = {}
        for roi_name, contour_list in predicted_contours.iteritems():
            slices_to_contours[image_ds.SOPInstanceUID][roi_name] = contour_list

    # Return mapping of slices to contours and ROI names to slices/contours
    return slices_to_contours

# A function to add to an RT
def to_rt():
    return False

# A function to create new from an RT
def from_rt():
    return False

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
