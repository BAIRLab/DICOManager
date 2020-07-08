import numpy as np
import os
import pydicom
import glob
import time
from dataclasses import dataclass
from beaunet_predict_dicom import _img_dims
import beaunet_predict_dicom as decon
import reconstruction as recon

'''
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
'''
rt_series = glob.glob('/home/eporter/eporter_data/optic_structures/dicoms/1010370/RTSTRUCT/*.dcm')
ct_series = glob.glob('/home/eporter/eporter_data/optic_structures/dicoms/1010370/CT/*.dcm')
rt = pydicom.dcmread(rt_series[0])
print(rt.RTROIObservationsSequence)
#rt.RTROIObservationsSequence = pydicom.sequence.Sequence([])
rt.RTROIObservationsSequence.clear()
print('empty')
print(rt.RTROIObservationsSequence)
rt.save_as('test.dcm')
new_rt = pydicom.dcmread('test.dcm')
print('Reading')
print(new_rt.RTROIObservationsSequence)
'''
ct_series.sort()
built = recon.struct(rt_series[0], wanted_contours=['skull'])
mask = []

uid_dict, ct_dict = decon._generate_dicts(ct_series)
for z in range(built.shape[-1]):
    mask.append(decon._array_to_coords_2D(built[0, :, :, z], ct_dict[z]))

rt = pydicom.dcmread(rt_series[0])
z_locs = []
for seq in rt.ROIContourSequence[0].ContourSequence:
    z_locs.append(seq.ContourData[2])

print(z_locs)
print(len(z_locs), len(mask))

for i, m in enumerate(mask):
    if m is not None:
        print(m[2])
    else:
        continue
'''