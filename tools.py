from __future__ import annotations
import abc
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import center_of_mass
import utils
import numpy as np
import pydicom
from skimage.transform import rescale
from typing import TYPE_CHECKING, Any, TypeVar, Union
from groupings import Cohort, FrameOfRef, Modality, ReconstructedVolume, ReconstructedFile

if TYPE_CHECKING:
    from groupings import Cohort, FrameOfRef, Modality, ReconstructedVolume, ReconstructedFile


# Custom Types
ReconVolumeOrFile = Union[ReconstructedVolume, ReconstructedFile]

# Tools are used to process reconstructed arrays
# DicomUtils are for DICOM reconstruction utilities
# Should probably unnest from the class and move to a seperate file


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


class ImgHandler:
    def pointer_conversion(self, img: ReconVolumeOrFile, path: str = None) -> ReconVolumeOrFile:
        """[Handles conversion of ReconstructedFile to Volume for any tools]

        Args:
            img (ReconVolumeOrFile): [Either ReconstructedFile or ReconstructedVolume]
            path (str, optional): [Path to write the modified files]. Defaults to None.

        Returns:
            ReconVolumeOrFile: [Returns modified image of same type as img]
        """
        if self._check_needed(img):
            utils.colorwarn('Will write over reconstructed volume files')
            if type(img) is ReconstructedFile:
                img.load_array()
                results = self._function(img)
                results.convert_to_pointer(path=path)
            else:
                results = self._function(img)
            return results
        return img

    @abc.abstractmethod
    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """[Determines if the tool should be applied to the volume]

        Args:
            img (ReconVolumeOrFile): [Image volume to apply tool]

        Returns:
            bool: [Determines if tool should be applied]

        Notes:
            This is an abstract method and each tool class requires
            implementation to work appropriately
        """
        return True


class WindowLevel(ImgHandler):
    def __init__(self, window, level):
        self.window = window
        self.level = level

    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        if img.is_struct:
            return img

        for name, volume in img.volumes.items():
            imgmin, imgmax = (volume.min(), volume.max())
            img.ImgAugmentations.wl_update(self.window, self.level, imgmin, imgmax)

            ctlo = self.level - self.window / 2
            cthi = ctlo + self.window

            temp = np.copy(volume)
            temp[temp < ctlo] = ctlo
            temp[temp > cthi] = cthi
            img.volumes[name] = temp

        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return img.Modality == 'CT'


class Normalize(ImgHandler):
    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        if img.is_struct:
            return img

        imgmin, imgmax = (None, None)
        for name, volume in img.volumes.items():
            if imgmin is None or imgmax is None:
                imgmin, imgmax = (volume.min(), volume.max())
                img.ImgAugmentations.norm_update(imgmin, imgmax)

            img.volumes[name] = (np.copy(volume) - imgmin) / (imgmax - imgmin)

        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return not img.ImgAugmentations.normalized


class Standardize(ImgHandler):
    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        if img.is_struct:
            return img

        for name, volume in img.volumes.items():
            imgmean = np.mean(volume)
            imgstd = np.std(volume)
            img.ImgAugmentations.std_update(imgmean, imgstd)

            img.volumes[name] = (np.copy(volume) - imgmean) / imgstd
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return not img.ImgAugmentations.standardized


class BiasFieldCorrection(ImgHandler):
    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
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
        img.ImgAugmentations.bias_corrected = True

        for name, volume in img.volumes.items():
            sitk_image = sitk.GetImageFromArray(volume)
            sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            output = corrector.Execute(sitk_image, sitk_mask)
            img.volumes[name] = sitk.GetArrayFromImage(output)

        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return img.Modality == 'MR'


class Interpolate(ImgHandler):
    """[Axial image interpolation function, currently using bilinear interpolation]

    Args:
        extrapolate (bool): [Specifies if extrapolation is permitted]. Defaults to False.

    Warnings:
        UserWarning: Warns if extrapolation is attempted but forbidden
        UserWarning: Warns that if path is not specified, interpolated images will be written
            over their existing reconstructed image volumes
    """
    def __init__(self, extrapolate=False):
        self.extrapolate = extrapolate

    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """[Actual interpolation function, obscured from user because class instance
            call should be used instead for more efficient performance]

        Args:
            img (ReconVolumeOrFile): [ReconstructedVolume or ReconstructedFile]

        Returns:
            ReconVolumeOrFile: [Modified image returned in same format as img input]
        """
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

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """[Checks if interpolation of the file is required]

        Args:
            img (ReconVolumeOrFile): [ReconstructedVolume or ReconstructedFile]

        Returns:
            bool: [Boolean if interpolation is needed]
        """
        empty = img.ImgAugmentations.empty_slices
        if not empty or len(empty) == 0:
            return False
        return True


# We could also resample by Field of View
class Resample(ImgHandler):
    def __init__(self, ratio: float = None, dims: list = None,
                 voxel_size: list = None, dz_limit: float = None,
                 dz_goal: float = None, dz_exact: float = None):
        """[Resamples volumetric image to dimensions, coordinates or voxel size]

        Args:
            ratio (float, optional): [Resampling ratio, either per-dimension or uniform
                with r>1 upsampling and r<1 downsampling]. Defaults to None.
            dims (list, optional): [Number of voxels per-dimension to resmaple to, with
                dimensions of None left unsampled. (e.g. [512, 512, None])]. Defaults to None.
            voxel_size (list, optional): [Voxel size, in mm, to resample image]. Defaults to None.
            dz_limit (float, optional): [Limited slice thickness, with dz > dz_limit being
                resampled to dz_goal or 1/2 slice thickness otherwise]. Defaults to None.
            dz_goal (float, optional): [Resampling goal if dz_limit is specified]. Defaults to None.
            dz_exact (float, optional): [Resamples all images to this exact dz, if
                specified. Will override dz_limit and dz_goal]. Defaults to None.

        Notes:
            Non-integer resampling ratios may cause artifacting errors. Therefore, resampling by
                factors of 2 is recommended, if at all possible.
            Using the dz_exact parameter may result in resampling a large number of image volumes,
                requiring significant computational resources and time to complete. Use sparingly.
        """
        self.ratio = ratio
        self.dims = dims
        self.voxel_size = voxel_size
        self.dz_limit = dz_limit
        self.dz_goal = dz_goal
        self.dz_exact = dz_exact
        message = 'Must specify either ratio or xy-dims'
        passes = self.ratio is not None or self.dims is not None
        assert passes, message

    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """[Downsample an image by a specified ratio]

        Args:
            img (ReconstructedVolume): [Image to resample]

        Returns:
            ReconstructedVolume: [Downsampled image array]

        Notes:
            TODO: Does not update voxel spacing yet
        """
        if self.ratio is not None:
            current_ratio = self.ratio
        elif self.dims is not None:
            current_ratio = self._dims_to_ratio(img)
        elif self.voxel_size is not None:
            current_ratio = self._voxel_size_to_ratio(img)
        else:
            raise TypeError('Need to specify a rescaling attribute')

        if self.dz_exact is not None or self.dz_limit is not None:
            current_ratio = self._dz_limit_to_ratio(img, previous=current_ratio)

        for name, volume in img.volumes.items():
            img.volumes[name] = rescale(np.copy(volume), current_ratio)

        if not img.ImgAugmentations.resampled:
            img.ImgAugmentations.resampled_update(img.dims.voxel_size, current_ratio)
            img.dims.resampled_update(current_ratio)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """[Determines if the volume or file requires resampling]

        Args:
            img (ReconVolumeOrFile): [Image volume or file]

        Returns:
            bool: [Requires resampling]

        Notes:
            Resampling heirarchy, where resampling occurs if any is True:
                1. If dz_exact is specified and dz is outside tolerance
                2. If dz_limit is specified and dz is outside tolerance
                3. If voxel_size is specified and img is outside tolerance
                4. If dims is specified and img is outside tolerance
                5. If not resampled previously
        """
        def condition(x, y):
            if x is None:
                return True
            return x == y

        if img.PatientID == '0933-638561':
            print(f'HERE!! - {img.dims.dz}')

        # slice thickness resampling
        if self.dz_exact is not None:
            if img.dims.dz != self.dz_exact:
                return True

        if self.dz_limit is not None:
            if img.dims.dz > self.dz_limit:
                return True

        # voxel resampling
        if self.voxel_size is not None:
            passes = [condition(x, y) for x, y in zip(self.voxel_size, img.dims.voxel_size)]
            return not all(passes)

        # xyz-dimensions resampling
        if self.dims is not None:
            passes = [condition(x, y) for x, y in zip(self.dims, img.dims.shape)]
            return not all(passes)

        return not img.ImgAugmentations.resampled

    def _compute_ratio(self, list1: list, list2: list) -> list:
        """[Computes resampling ratio between two lists]

        Args:
            list1 (list): [Desired parameters]
            list2 (list): [Current parameters]

        Returns:
            list: [Resampling ratio]
        """
        ratio = []
        for x0, x1 in zip(list1, list2):
            if x0 is None:
                x0 = x1
            ratio.append(x0 / x1)
        return ratio

    def _dims_to_ratio(self, img: ReconVolumeOrFile) -> list:
        """[Calculates the resampling ratio based on dimensions]

        Args:
            img (ReconVolumeOrFile): [Image volume or file]

        Returns:
            list: [Resampling ratio]
        """
        return self._compute_ratio(self.dims, img.dims.shape)

    def _voxel_size_to_ratio(self, img: ReconVolumeOrFile) -> list:
        """[Calculates the resampling ratio based on voxel size]

        Args:
            img (ReconVolumeOrFile): [Image volume or file]

        Returns:
            list: [Resampling ratio]
        """
        return self._compute_ratio(self.voxel_size, img.dims.voxel_size)

    def _dz_limit_to_ratio(self, img: ReconVolumeOrFile, previous: list) -> list:
        """[Calculates the z-axis resampling ratio based on slice thickness limit]

        Args:
            img (ReconVolumeOrFile): [Image volume or file]
            prevoius: (list): [Previous list of resampling ratios]

        Returns:
            list: [Resampling ratio]
        """
        if self.dz_exact is not None:
            if img.dims.dz != self.dz_exact:
                previous[-1] = img.dims.dz / self.dz_exact
        elif img.dims.dz > self.dz_limit:
            print(f'{img.PatientID} {img.dims.dz}, {previous}')
            if self.dz_goal is not None:
                previous[-1] = img.dims.dz / self.dz_goal
            else:
                previous[-1] = 2
                current = img.dims.dz / previous[-1]
                while current > self.dz_limit:
                    previous[-1] += 2
                    current = img.dims.dz / previous[-1]
            print(f'{img.PatientID} {previous}')
        return previous


class Crop(ImgHandler):
    def __init__(self, crop_size, centroid=None, structure=None,
                 offset=None, custom_centroid=None):
        # Custom centroid takes ReconstructedVolume and returns a list of
        # ints with dimensions equal to the image volume dimensions
        self.crop_size = crop_size
        self.centroid = centroid
        self.custom_centroid = custom_centroid
        self.structure = structure
        self.offset = offset
        self._centroids = {}

    def __call__(self, img, path):
        return self.pointer_conversion(img, path)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """[Crops an image volume and updates headers accordingly]

        Args:
            img (ReconVolumeOrFile): [Image volume object to crop]

        Returns:
            ReconVolumeOrFile: [description]

        Notes:
            Centroid will not be observed if the cropped volume will be
            smaller than the crop_size. Will shift centroid to maintain
            the specific crop size
            This function currently does not update the ReconstructedVolume header
            or ReconstructedVolume.dicom_header to reflect the new dimensions
        """
        # Need to update the VolumeDimensions header and dicom header too
        imgshape = np.array(img.dims.shape)
        img_coords = np.zeros((2, len(imgshape)))
        patient_coords = np.zeros((2, len(imgshape)))

        if self.centroid is None:
            this_centroid = self._calc_centroid(imgshape)
        elif self.custom_centroid is not None:
            this_centroid = self.custom_centroid(img)
        else:
            this_centroid = self.centroid

        for i, (point, size) in enumerate(zip(this_centroid, self.crop_size)):
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
        img.ImgAugmentations.crop_update(img_coords, patient_coords)
        return img

    def _calc_centroid(self, img):
        if self.structure is None:
            return np.array(img.dims.shape) // 2

        frame = img.ancestors[-1]
        if frame.name in self._centroids:
            return self._centroids[frame.name]

        for mod in frame.iter_modalities:
            if mod.name == 'RTSTRUCT':
                for name, volume in mod.volumes_data.items():
                    if name in self.structure:
                        centroid = center_of_mass(volume)
                        self._centroids.update({frame.name: centroid})
                        return centroid

        raise ValueError(f'Structure not found for patient {img.PatientID}')

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        passes = [x == y for x, y in zip(self.crop_size, img.dims.shape)]
        return not all(passes)


"""
Start of Code for the interpolation of RTSTRUCTs. These will require more user specification to prevent
the interpolation of intentional gaps within the structure sets.

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
