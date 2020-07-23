import hashlib
import numpy as np
import os
import pydicom
import random
import uuid
import scipy
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy import spatial
from skimage import measure as skm


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


def wire_mask(arr, invert=False):
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
    assert arr.ndim == 2, 'The input boolean mask is not 2D'

    if arr.dtype != 'bool':
        arr = np.array(arr, dtype='bool')
    if invert:
        return binary_dilation(arr) ^ arr
    return binary_erosion(arr) ^ arr


def sort_points(points, method='kd'):
    """
    Function
    ----------
    Given a set of points, the points are sorted by the
        specified method and returned
    
    Parameters
    ----------
    points : numpy.ndarray
        A Nx3 numpy array of (x, y, z) coordinates
    method : str (Default = 'kd')
        The method to use for sorting. Options are:
            'kd' : Sort by KDTrees, convex robust
            'ccw' : Sort counterclockwise, not convex robust
            'cw' : Sort clockwise, not convex robust
    
    Returns
    ----------
    numpy.ndarray
        The points in the provided array, sorted as specified
    """
    if method == 'kd':
        return kd_sort_nearest(points)
    elif method == 'ccw':
        return sort_points_ccw(points)
    elif method == 'cw':
        return sort_points_ccw(points, counterclockwise=False)
    else:
        raise TypeError('The method must be one of kd, cw, ccw')


def sort_points_ccw(points, counterclockwise=True):
    """
    Function
    ----------
    Computes the counterclockwise angle between CoM to point and y-axis

    Parameters
    ----------
    points : numpy.ndarray
        A Nx3 numpy array of (x, y, z) coordinates
    
    Returns
    ----------
    numpy.ndarray
        The points array sorted in a counterclockwise direction
    
    Notes
    ----------
    Sorting (counter)clockwise only works for concave structures, for
        convex structures, use kd_sort_nearest
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


def kd_sort_nearest(points):
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


def get_first_index(points):
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
        quicker than fisrt clockwise sorting all points
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


def find_nearest(inner, outer):
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


def merge_sorted_points(inner, outer):
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
    Because we append to merged for each inner, when larger we have recursed 
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
        previous = [] # for index offset due to previous insertions 
        for n_inner in inner:
            point_pair, index_pair = find_nearest(n_inner, og_outer)
            n_inner = np.roll(n_inner, n_inner.shape[0] - index_pair[0], axis=0)
            n_inner = np.append(n_inner, point_pair, axis=0)
            offset = calc_offset(previous, index_pair[1])
            outer = np.insert(outer, index_pair[1] + offset + 1, n_inner, axis=0)
            previous.append((index_pair[1], n_inner.shape[0]))
    else:
        point_pair, index_pair = find_nearest(inner, outer)
        inner = np.roll(inner, inner.shape[0] - index_pair[0], axis=0)
        inner = np.append(inner, point_pair, axis=0)
        outer = np.insert(outer, index_pair[1] + 1, inner, axis=0)
    return outer


def calc_offset(prevoius, index):
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
        An integer represnting the number of slices to offset the insertion

    Notes
    ----------
    When we have mulitple nested structures, we need to compute what the 
        potential offset is when placing into the sorted list. We cannot do
        this in the original list because then future structures may map to 
        a newly inserted structure as opposed to the exterior polygon 
    """
    offset = 0
    for item in prevoius:
        idx, n_pts = item
        if index > idx:
            offset += n_pts
    return offset


def split_by_holes(poly):
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


def all_points_merged(poly, merged):
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
    Because we append to merged for each inner, when larger we have recursed 
        through all the points, unless the holes constitute a single pixel,
        in which they are likely undesired anyways
    """
    mask = wire_mask(poly)
    total_pts = np.rollaxis(np.transpose(mask.nonzero()), 0, 2)
    return total_pts.shape[0] <= merged.shape[0]


def poly_to_coords_2D(poly, ctcoord, flatten=True, invert=False):
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
    Invert is called as the negation of the prevoius value. This is
        done in the case where a nested hole contains a polygon and
        the edge detection must alternate between erison and dilation
    """
    # Check if the polygon has hole and inner is None if no holes
    inner, outer = split_by_holes(poly)
    if inner is not None:
        inner_pts = poly_to_coords_2D(
            inner, ctcoord, flatten=False, invert=not invert)
        outer_pts = poly_to_coords_2D(
            outer, ctcoord, flatten=False, invert=invert)
        merged = merge_sorted_points(inner_pts, outer_pts)
        if all_points_merged(poly, merged) and not invert:
            return merged.flatten()
        return merged

    # Check if there are mupltiple polygons
    polygons, n_polygons = skm.label(poly, connectivity=2, return_num=True)
    if n_polygons > 1:
        i_polys = [polygons == n for n in range(
            1, n_polygons + 1)]  # 0=background
        return [poly_to_coords_2D(x, ctcoord, flatten=False, invert=True) for x in i_polys]

    # Convert poly to coords and sort
    mask = wire_mask(poly, invert=invert)
    points = np.rollaxis(
        ctcoord[tuple(mask.nonzero())+np.index_exp[:]].T, 0, 2)
    points_sorted = sort_points(points)
    if flatten:
        return points_sorted.flatten()
    return points_sorted


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
