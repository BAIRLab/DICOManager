import hashlib
import numpy as np
import os
import pydicom
import random
import uuid
from scipy.ndimage.morphology import binary_erosion


def prepare_coordinate_mapping(ct_hdr):
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


def wire_mask(arr):
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
    Another option is to use VTK:
        https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/
    Either way, the reconstruction polygon fill is first priority, and then
        this deconstruction minium polygon surface computation.
    """
    # TODO: #10 While this returns the surface, the vertex ordering is incorrect. We need to use ITK's ContourExtractor2DImageFilter function
    assert arr.ndim == 2, 'The input boolean mask is not 2D'

    if arr.dtype != 'bool':
        arr = np.array(arr, dtype='bool')

    return binary_erosion(arr) ^ arr


def sort_points(points, counterclockwise=True):
    """
    Function
    ----------
    Computes the counterclockwise angle between CoM to point and y-axis

    Parameters
    ----------
    points: numpy.ndarray
        A Nx2 numpy array of (x, y) coordinates
    
    Returns
    ----------
    numpy.ndarray
        The points array sorted in a counterclockwise direction
    """
    CoM = np.mean(points, axis=0)

    def _angle(v1_og):
        v1 = v1_og - CoM
        v2 = np.array([0, -CoM[1], CoM[2]])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        if counterclockwise:
            if v1_og[0] > CoM[0]:
                return np.arccos(np.dot(v1_u, v2_u))
            return 100 - np.arccos(np.dot(v1_u, v2_u))
        # Clockwise sorting
        if v1_og[0] < CoM[0]:
            return -np.arccos(np.dot(v1_u, v2_u))
        return 100 + np.arccos(np.dot(v1_u, v2_u))

    return np.array(sorted(points, key=_angle))


# DICOM stores each polygon as a unique item in the ContourImageSequence
def poly_to_coords_2D(poly, ct_hdr, flatten=True):
    """
    Function
    ----------
    Given 2D boolean array, eturns an array of surface coordinates in
        the patient coordinate system

    Parameters
    ----------
    ct_hdr : pydicom.dataset.FileDataset
        A CT dicom object to compute the image coordinate locations upon
        where 'hdr' means that the PixelData field is not required
    poly : numpy.ndarray
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
    assert poly.ndim == 2, 'The input boolean mask is not 2D'
    assert type(
        ct_hdr) is pydicom.dataset.FileDataset, 'ct_dcm is not a pydicom dataset'

    # If there is no contour on the slice, return no points
    if not np.sum(poly):
        return None

    mask = wire_mask(poly)
    coords = prepare_coordinate_mapping(ct_hdr)
    points = np.rollaxis(
        coords[tuple(mask.nonzero()) + np.index_exp[:]].T, 0, 2)
    points_sorted = sort_points(points)

    if not flatten:
        return points_sorted
    return points_sorted.flatten()


# TODO: #8 Move common utilities into a shared library for re and deconstruction
def img_dims(dicom_list):
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
    # Build dict of instance num -> location
    int_list = []
    loc_list = []

    for f in dicom_list:
        dcm = pydicom.dcmread(f, stop_before_pixels=True)
        int_list.append(round(dcm.InstanceNumber))
        loc_list.append(float(dcm.SliceLocation))

    # Sort both lists based on the int_list ordering
    int_list, loc_list = map(np.array, zip(*sorted(zip(int_list, loc_list))))

    # Calculate slice thickness
    loc0, loc1 = loc_list[:2]
    inst0, inst1 = int_list[:2]
    thickness = abs((loc1-loc0) / (inst1-inst0))

    # Compute if Patient and Image coords are flipped relatively
    flip = False if loc0 > loc1 else True  # Check

    # Compute number of slices and account for missing dicom files
    n_slices = round(1 + (loc_list.max() - loc_list.min()) / thickness)

    # Accounts for known missing instances on the low end. This is likely
    # unnecessary but good edge conditon protection for missing slices
    # Probably could save runtime by rewriting to use arrays
    if int_list.min() > 1:
        diff = int_list.min() - 1
        int_list, loc_list = list(int_list), list(loc_list)
        n_slices += diff
        int_list += list(range(1, diff + 1))
        # Adds the new locations to correspond with instances
        if flip:
            loc_list += list(loc0 - np.arange(1, diff + 1) * thickness)
        else:
            loc_list += list(loc0 + np.arange(1, diff + 1) * thickness)
        # Resorts the list with the new points
        int_list, loc_list = map(np.array, zip(
            *sorted(zip(int_list, loc_list))))

    return thickness, n_slices, loc_list.min(), loc_list.max(), flip


# TODO: Check with MIM on how they generate their Instance UID
def generate_instance_uid():
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
    # Modified from pydicom.uid.generate_uid
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
