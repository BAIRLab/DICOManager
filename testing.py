import numpy as np
import os
import pydicom
import glob
import time

dcm = pydicom.dcmread(glob.glob('/home/eporter/eporter_data/hippo_data/1112686/CT/*.dcm')[0])

def evan(dcm):
    """
    Function
    ----------
    Given a DICOM CT image slice, returns an array of pixel coordinates

    Parameters
    ----------
    dcm : pydicom.dataset.FileDataset
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
    X_x, X_y, X_z = np.array(dcm.ImageOrientationPatient[:3]).T
    Y_x, Y_y, Y_z = np.array(dcm.ImageOrientationPatient[3:]).T
    S_x, S_y, S_z = np.array(dcm.ImagePositionPatient)
    D_j, D_i = np.array(dcm.PixelSpacing)
    j, i = np.indices((dcm.Rows, dcm.Columns))

    M = np.array([[X_x*D_i, Y_x*D_j, 0, S_x],
                  [X_y*D_i, Y_y*D_j, 0, S_y],
                  [X_z*D_i, Y_z*D_j, 0, S_z],
                  [0, 0, 0, 1]])

    C = np.array([i, j, 0, 1])

    # Returns coordinates in [x, y, 3]
    return np.rollaxis(np.stack(np.dot(M[:-1], C)), 0, 3)


def leonid(image_slice):
    """
    Given a DICOM CT image slice, returns a numpy array with the coordinates of each pixel.
    :param image_slice: A pydicom dataset representing a CT slice in DICOM format.
    :return: A numpy array of shape Mx2 where M is image_slice.rows x image_slice.cols, the number of(x,y) pairs
             representing coordinates of each pixel.
    """
    M = np.array(
        [[np.array(image_slice.ImageOrientationPatient[0:3])[0] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[0] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[0]],

         [np.array(image_slice.ImageOrientationPatient[0:3])[1] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[1] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[1]],

         [np.array(image_slice.ImageOrientationPatient[0:3])[2] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[2] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[2]],

         [0, 0, 0, 1]
         ])

    pixel_coord_array = np.zeros(
        (image_slice.Rows, image_slice.Columns), dtype=(float, 3))

    pixel_idx_array = np.indices((image_slice.Rows, image_slice.Columns))

    it = np.nditer(op=[pixel_idx_array[0],  # Array of pixel row indices (j)
                       pixel_idx_array[1],  # Array of pixel col indices (i)
                       # Output array of pixel x coords
                       pixel_coord_array[:, :, 0],
                       # Output array of pixel y coords
                       pixel_coord_array[:, :, 1],
                       pixel_coord_array[:, :, 2]],  # Output array of pixel z coords
                   flags=['external_loop', 'buffered'],
                   op_flags=[['readonly'],
                             ['readonly'],
                             ['writeonly', 'no_broadcast'],
                             ['writeonly', 'no_broadcast'],
                             ['writeonly', 'no_broadcast']])

    for (j, i, Px, Py, Pz) in it:
        C = np.array([i, j, 0, 1])
        P = np.dot(M, C)

        Px[...] = P[0]
        Py[...] = P[1]
        Pz[...] = P[2]

    return pixel_coord_array

test1 = leonid(dcm)
test2 = evan(dcm)

np.testing.assert_almost_equal(test1, test2)

rands = np.random.randint(2, size=(512, 512))

def mask_to_coords(array, coords):
    return coords[tuple(array.nonzero()) + np.index_exp[:]].T

print(np.sum(rands))
print(mask_to_coords(rands, test2))
print(mask_to_coords(rands, test2).flatten().shape)

'''
start = time.time()
for _ in range(10000):
    _ = leonid(dcm)
end = time.time()
totl = end - start

start = time.time()
for _ in range(10000):
    _ = evan(dcm)
end = time.time()
tote = end - start

print(tote / totl)
'''