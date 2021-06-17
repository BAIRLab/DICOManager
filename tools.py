from __future__ import annotations
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
import utils
import numpy as np
import pydicom
from skimage.transform import rescale
from typing import TYPE_CHECKING
from groupings import Cohort, FrameOfRef, Modality, ReconstructedVolume, ReconstructedFile

if TYPE_CHECKING:
    from groupings import Cohort, FrameOfRef, Modality, ReconstructedVolume, ReconstructedFile

# Tools are used to process reconstructed arrays
# DicomUtils are for DICOM reconstruction utilities
# Should probably unnest from the class and move to a seperate file

def handle_pointers(func):
    def wrapped(img, *args, **kwargs):
        if type(img) is ReconstructedFile:
            utils.colorwarn('Will write over reconstructed volume files')
            img.load_array()
            results = func(img, *args, **kwargs)
            results.convert_to_pointer()
        else:
            results = func(img, *args, **kwargs)
        return results
    return wrapped


@handle_pointers
def dose_max_points(dose_array: np.ndarray,
                    dose_coords: np.ndarray = None) -> np.ndarray:
    """[Calculates the dose maximum point in an array, returns index or coordinates]

    Args:
        dose_array (np.ndarray): [A reconstructed dose array]
        dose_coords (np.ndarray, optional): [Associated patient coordinates]. Defaults to None.

    Returns:
        np.ndarray: [The dose max index, or patient coordinates, if given]
    """
    index = np.unravel_index(np.argmax(dose_array), dose_array.shape)

    if dose_coords:
        return dose_coords[index]
    return index


@handle_pointers
def window_level(img: ReconstructedVolume, window: int,
                 level: int) -> ReconstructedVolume:
    """[Applies a window and level to the given object. Works for either HU or CT number,
        whichever was specified during reconstruction of the array]

    Args:
        img (ReconstructedVolume): [Image volume object to window and level]
        window (int): [window in either HU or CT number]
        level (int): [level in either HU or CT number]

    Returns:
        ReconstructedVolume: [Image volume with window and level applied]
    """
    imgmin, imgmax = (img.min(), img.max())
    img.augmetnations.wl_update(window, level, imgmin, imgmax)

    ctlo = level - window / 2
    cthi = ctlo + window

    temp = np.copy(img.array)
    temp[temp < ctlo] = ctlo
    temp[temp > cthi] = cthi
    img.array = temp
    return img


@handle_pointers
def normalize(img: ReconstructedVolume) -> ReconstructedVolume:
    """[Normalizes an image volume ojbect]

    Args:
        img (ReconstructedVolume): [The image volume object to normalize]

    Returns:
        ReconstructedVolume: [Normalized image volume object]
    """
    imgmin, imgmax = (img.array.min(), img.array.max())
    img.augmentations.norm_update(imgmin, imgmax)

    temp = (np.copy(img.array) - imgmin) / (imgmax - imgmin)
    img.array = temp
    return img


@handle_pointers
def standardize(img: ReconstructedVolume) -> ReconstructedVolume:
    """[Standardizes an image volume object]

    Args:
        img (ReconstructedVolume): [The image volume object to standardize]

    Returns:
        ReconstructedVolume: [Standardized image volume object]
    """
    imgmean = np.mean(img.array)
    imgstd = np.std(img.array)
    img.augmentation.std_update(imgmean, imgstd)

    temp = (np.copy(img.array) - imgmean) / imgstd
    img.array = temp
    return img


@handle_pointers
def crop(img: ReconstructedVolume, centroid: np.ndarray,
         crop_size: np.ndarray) -> ReconstructedVolume:
    """[Crops an image volume and updates headers accordingly]

    Args:
        img (ReconstructedVolume): [Image volume object to crop]
        centroid (np.ndarray): [central cropping value]
        crop_size (np.ndarray): [dimensions of final cropped array]

    Returns:
        ReconstructedVolume: [description]

    Notes:
        Centroid will not be observed if the cropped volume will be
        smaller than the crop_size. Will shift centroid to maintain
        the specific crop size
        This function currently does not update the ReconstructedVolume header
        or ReconstructedVolume.dicom_header to reflect the new dimensions
    """
    # Need to update the VolumeDimensions header and dicom header too
    imgshape = img.array.shape()
    img_coords = np.zeros((2, len(imgshape)))
    patient_coords = np.zeros((2, len(imgshape)))

    for i, (point, size) in enumerate(zip(centroid, crop_size)):
        low = min(0, point - size // 2)
        high = low + size
        if high > (imgshape[i] - 1):
            high = (imgshape[i] - 1)
            low = high - size
            img_coords[0, i] = low
            img_coords[1, i] = high

    coordrange = img.dims.coordrange()
    for i, (low, high) in enumerate(img_coords.T):
        patient_coords[0, i] = coordrange[i, low]
        patient_coords[1, i] = coordrange[i, high]

    xlo, xhi, ylo, yhi, zlo, zhi = img_coords.T.flatten()
    temp = np.copy(img.array)
    img.array = temp[xlo: xhi, ylo: yhi, zlo: zhi]

    # img.crop_update()
    img.augmentations.crop_update(img_coords, patient_coords)
    return img


@handle_pointers
def resample(img: ReconstructedVolume, ratio: float) -> ReconstructedVolume:
    """[Downsample an image by a specified ratio]

    Args:
        img (ReconstructedVolume): [Image to resample]
        ratio (float): [Ratio to downsample]

    Returns:
        ReconstructedVolume: [Downsampled image array]

    Notes:
        TODO: Does not update voxel spacing yet
    """
    temp = np.copy(img.array)
    img.array = rescale(temp, ratio)
    img.augmentations.resample_update(None, round(ratio))
    return img


@handle_pointers
def bias_field_correction(self, img: ReconstructedVolume) -> ReconstructedVolume:
    """[MRI Bias Field Correction]

    Args:
        img (ReconstructedVolume): [MRI Image Volume to be N4 bias corrected]

    Returns:
        ReconstructedVolume: [N4 bias corrected image]

    Notes:
        SimpleITK is not compiled for PowerPC, therefore this function is not
        avaliable to all architetures
    """
    import SimpleITK as sitk
    img.augmentations.bias_corrected = True

    sitk_image = sitk.GetImageFromArray(img.array)
    sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(sitk_image, sitk_mask)
    img.array = sitk.GetArrayFromImage(output)

    return img


@handle_pointers
def interpolate(img: ReconstructedVolume,
                extrapolate: bool = False) -> ReconstructedVolume:
    # Simple linear interpolation, will use more sophisticated bilinear interpolation
    # in the second revision
    empty_slices = img.ImgAugmentations.empty_slices
    if empty_slices and not img.is_struct:
        interpolated = []
        extrapolated = []
        for name, volume in img.volumes.items():
            for eslice in empty_slices:
                zlo, zhi = (1, 1)
                lo_slice, hi_slice = (None, None)
                # Find filled slice below
                while (eslice - zlo) > 0:
                    lo_slice = volume[..., eslice-zlo]
                    if lo_slice.size:
                        break
                    zlo += 1
                # Find filled slice above
                while (eslice + zhi) < (img.dims.slices - 1):
                    hi_slice = volume[..., eslice+zhi]
                    if hi_slice.size:
                        break
                    zhi += 1
                # interpolate or extrapolate, if valid
                if lo_slice is not None and hi_slice is not None:
                    volume[..., eslice] = (hi_slice + lo_slice) / 2
                    interpolated.append(eslice)
                elif not extrapolate:
                    utils.colorwarn(f'Cannot interpolate {img.PatientID}, try extroplate=True')
                elif lo_slice is not None:
                    volume[..., eslice] = hi_slice
                else:
                    volume[..., eslice] = lo_slice

        interpolated.sort()
        img.ImgAugmentations.interpolated_update(interpolated, extrapolated)
    return img


class handler:
    def pointer_conversion(self, img, path):
        if self._check_needed(img):
            utils.colorwarn('Will write over reconstructed volume files')
            if type(img) is ReconstructedFile:
                img.load_array()
                print(img.PatientID, img.ImgAugmentations.interpolated)
                results = self._function(img)
                print(img.PatientID, results.ImgAugmentations.interpolated)
                results.convert_to_pointer(output=True, path=path)
            else:
                results = self._function(img)
            return results
        return img


class Interpolate(handler):
    def __init__(self, extrapolate=False):
        self.extrapolate = extrapolate

    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img):
        # Simple linear interpolation, will use more sophisticated bilinear interpolation
        # in the second revision
        empty_slices = img.ImgAugmentations.empty_slices
        if empty_slices and not img.is_struct:
            interpolated = []
            extrapolated = []
            for name, volume in img.volumes.items():
                for eslice in empty_slices:
                    zlo, zhi = (1, 1)
                    lo_slice, hi_slice = (None, None)
                    # Find filled slice below
                    while (eslice - zlo) > 0:
                        lo_slice = volume[..., eslice-zlo]
                        if lo_slice.size:
                            break
                        zlo += 1
                    # Find filled slice above
                    while (eslice + zhi) < (img.dims.slices - 1):
                        hi_slice = volume[..., eslice+zhi]
                        if hi_slice.size:
                            break
                        zhi += 1
                    # interpolate or extrapolate, if valid
                    if lo_slice is not None and hi_slice is not None:
                        volume[..., eslice] = (hi_slice + lo_slice) / 2
                        interpolated.append(eslice)
                    elif not self.extrapolate:
                        utils.colorwarn(f'Cannot interpolate {img.PatientID}, try extroplate=True')
                    elif lo_slice is not None:
                        volume[..., eslice] = hi_slice
                    else:
                        volume[..., eslice] = lo_slice

            interpolated.sort()
            img.ImgAugmentations.interpolated_update(interpolated, extrapolated)
        return img

    def _check_needed(self, img):
        empty = img.ImgAugmentations.empty_slices
        if not empty or len(empty) == 0:
            return False
        return True


"""
@handle_pointers
def interpolate_contour(img: ReconstructedVolume, extrapolate=False) -> ReconstructedVolume:
    # https://stackoverflow.com/questions/48818373/interpolate-between-two-images
    empty_slices = img.ImgAugmentations.empty_slices
    if empty_slices and img.is_struct:
        interpolated = []
        extrapolated = []
        for name, volume in img.volumes.items():
            for eslice in empty_slices:
                zlo, zhi = (1, 1)
                lo_slice, hi_slice = (None, None)
                # Find filled slice below
                while (eslice - zlo) > 0:
                    lo_slice = volume[..., eslice-zlo]
                    if lo_slice.size:
                        break
                    zlo += 1
                # Find filled slice above
                while (eslice + zhi) < (img.dims.slices - 1):
                    hi_slice = volume[..., eslice+zhi]
                    if hi_slice.size:
                        break
                    zhi += 1
                # interpolate or extrapolate, if valid
                if lo_slice.size and hi_slice.size:
                    volume[..., eslice] = (hi_slice + lo_slice) / 2
                    interpolated.append(eslice)
                elif not extrapolate:
                    utils.colorwarn(f'Cannot interpolate {img}, try extroplate=True')
                elif not lo_slice.size:
                    volume[..., eslice] = hi_slice
                else:
                    volume[..., eslice] = lo_slice
    return interp_shape(lo_slice, hi_slice, 0.5)



def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    from mahotas import bwperim
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im


def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im


def interp_shape(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision > 2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
    xi = np.c_[np.full((r**2), precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out
"""
