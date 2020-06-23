import numpy as np
import scipy
from scipy.ndimage.morphology import binary_erosion, binary_dilation

def WireMask(array):
    array = np.array(array, dtype='bool')
    return binary_erosion(array) ^ array

def SurfCoords2D(array, pat_coords=True):
    coords = np.transpose(WireMask(array).nonzero())
    if pat_coords:
        return ImgToPatCoords
    return coords

def SurfCoords3D(array, pat_coords=True):
    fill = []
    for z in range(array.shape[0]):
        fill.append(SurfCoords2D(array[z]))
    if pat_coords:
        return ImgToPatCoords(fill)
    return fill

def ImgToPatCoords(coords, voxel_size, origin):
    return (coords * voxel_size) + origin

test_array = np.zeros((3, 5, 5))
test_array[0:2, 1:4, 1:4] = 1
print(SurfCoords3D(test_array))
print(test_array)
print(WireMask(test_array))
print(binary_erosion(test_array))

# Name is stored at dcm.StructureSetROISequence.ROIName
# List of contour slice points is stored at dcm.StructureSetROISequence[index].ContourSequence
