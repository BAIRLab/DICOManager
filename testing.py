import numpy as np
import os
import pydicom
import glob
import time
from dataclasses import dataclass
'''

@dataclass
class UidItem:
    hdr: pydicom.dataset.Dataset()
    ct_thick: float
    ct_loc0: float
    uid: str = None
    loc: int = None
    ct_store: pydicom.dataset.Dataset() = None

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def _get_z_loc(self):
        diff = self.ct_loc0 - self.hdr.SliceLocation
        return int(round(abs(diff / self.ct_thick)))

    def __post_init__(self):
        self.loc = self._get_z_loc()
        self.uid = self.hdr.SOPInstanceUID
        self.ct_store = pydicom.dataset.Dataset()
        self.ct_store.ReferencedSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')
        self.ct_store.ReferencedSOPInstanceUID = self.hdr.SOPInstanceUID
        self.hdr = None


# Could make this entire dictionary into a dataclass ... will need to evaluate it
def _generate_uid_dict(ct_series):
    uid_dict = {}
    ct_thick, _, ct_loc0, _, _ = _img_dims(ct_series)
    for ct in ct_series:
        hdr = pydicom.dcmread(ct, stop_before_pixels=True)
        uid_dict.update({hdr.SOPInstanceUID: UidItem(hdr, ct_thick, ct_loc0)})
    return dict(sorted(uid_dict.items()))


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
    
    if int_list.min() > 1:
        diff = int_list.min() - 1
        # Probably could save runtime by rewriting to use arrays
        int_list, loc_list = list(int_list), list(loc_list)
        n_slices += diff
        int_list += [*range(1, diff + 1)]
        if flip:
            loc_list += list(loc0 - np.arange(1, diff + 1) * thickness)
        else:
            loc_list += list(loc0 + np.arange(1, diff + 1) * thickness)
        
        int_list, loc_list = map(np.array, zip(*sorted(zip(int_list, loc_list))))

    return thickness, n_slices, loc_list.min(), loc_list.max(), flip


import itk
import SimpleITK as sitk
import reconstruction as recon
from scipy.spatial import Delaunay
import numpy as np
from scipy.ndimage.morphology import binary_erosion

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

rt_series = glob.glob('/home/eporter/eporter_data/optic_structures/dicoms/1010370/RTSTRUCT/*.dcm')
ds = pydicom.dcmread(rt_series[0])
names = [x.ROIName for x in ds.StructureSetROISequence]
rts = recon.struct(rt_series[0], names[:1])
new = rts[0, :, :, 40]
surf = new ^ binary_erosion(new)
test = np.transpose(surf.nonzero())
out = alpha_shape(test, alpha= 1.65)
print(test)
print(out)
'''

import numpy as np

def ccw_sort(points):
    CoM = np.mean(points, axis=0)
    
    def _angle(v1):
        v1 = np.array([0, -CoM[1]]) 
        v2 = v1 - CoM
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        if v1[0] > CoM[1]:
            return np.arccos(np.dot(v1_u, v2_u))
        return 10 - np.arccos(np.dot(v1_u, v2_u))
    
    return np.array(sorted(points, key=_angle))

i_sort = ccw_sort(indx)
print(i_sort)