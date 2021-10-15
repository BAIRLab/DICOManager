from __future__ import annotations
import abc
import multiprocessing
import numpy as np
from anytree import NodeMixin
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from scipy.ndimage import zoom
from typing import Union
from . import utils
from .groupings import ReconstructedVolume, ReconstructedFile, FrameOfRef

# Custom Types
ReconVolumeOrFile = Union[ReconstructedVolume, ReconstructedFile]

# Tools are used to process reconstructed arrays
# DicomUtils are for DICOM reconstruction utilities
# Should probably un-nest from the class and move to a separate file


class ImgHandler:
    def __call__(self, img, path):
        return self.file_conversion(img, path)

    def file_conversion(self, img: ReconVolumeOrFile, path: str = None) -> ReconVolumeOrFile:
        """Handles conversion of ReconstructedFile to Volume for any tools

        Args:
            img (ReconVolumeOrFile): Either ReconstructedFile or ReconstructedVolume
            path (str, optional): Path to write the modified files. Defaults to None.

        Returns:
            ReconVolumeOrFile: Returns modified image of same type as img
        """
        if self._check_needed(img):
            utils.colorwarn('Will write over reconstructed volume files')
            if type(img) is ReconstructedFile:
                img.load_array()
                img = self._function(img)
                img.convert_to_file(path=path)
            else:
                img = self._function(img)
            return img
        return img

    @abc.abstractmethod
    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """Determines if the tool should be applied to the volume

        Args:
            img (ReconVolumeOrFile): Image volume to apply tool

        Returns:
            bool: Determines if tool should be applied

        Notes:
            This is an abstract method and each tool class requires
            implementation to work appropriately
        """
        return True


class WindowLevel(ImgHandler):
    """Applies window and level to the CT image volume

    Args:
        window (int): Window width
        level (int): Level value

    Notes:
        Default reconstruction of the images yields volumes with
            values of HU. Window and level values should correspond to
            HU unless normalization / standardization has been previously
            applied.
    """
    def __init__(self, window, level):
        self.window = window
        self.level = level

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
    """Normalize the image volume
    """
    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        if img.is_struct:
            return img

        imgmin, imgmax = (None, None)
        for name, volume in img.volumes.items():
            if imgmin is None or imgmax is None:
                imgmin, imgmax = (volume.min(), volume.max())
                img.ImgAugmentations.norm_update(imgmin, imgmax)

            if imgmin == imgmax and imgmin != 0:
                img.volumes[name] = volume / imgmin
            elif imgmin != imgmax:
                img.volumes[name] = (volume - imgmin) / (imgmax - imgmin)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return not img.ImgAugmentations.normalized


class Standardize(ImgHandler):
    """Standardize the image volume
    """
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
    def __init__(self, histogram_bins: int = 200):
        self.histogram_bins = histogram_bins

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """MRI N4 Bias Field Correction

        Args:
            img (ReconVolumeOrFile): MRI Image Volume to be N4 bias corrected

        Returns:
            ReconstructedVolume: N4 bias corrected image

        Notes:
            SimpleITK is not compiled for PowerPC, therefore this function is not
            avaliable to all architetures
        """
        import SimpleITK as sitk
        img.ImgAugmentations.bias_corrected = True

        for name, volume in img.volumes.items():
            sitk_image = sitk.GetImageFromArray(volume)
            sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, self.histogram_bins)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            output = corrector.Execute(sitk_image, sitk_mask)
            img.volumes[name] = sitk.GetArrayFromImage(output)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        return img.Modality == 'MR'


class Interpolate(ImgHandler):
    """Axial image interpolation function, currently using bilinear interpolation

    Warnings:
        UserWarning: Warns if extrapolation is attempted but forbidden
        UserWarning: Warns that if path is not specified, interpolated images will be written
            over their existing reconstructed image volumes
    """
    def __init__(self, extrapolate=False):
        self.extrapolate = extrapolate

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """Actual interpolation function, obscured from user because class instance
            call should be used instead for more efficient performance

        Args:
            img (ReconVolumeOrFile): ReconstructedVolume or ReconstructedFile

        Returns:
            ReconVolumeOrFile: Modified image returned in same format as img input
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
                    elif hi_slice is not None:
                        volume[..., eslice] = hi_slice
                    else:
                        volume[..., eslice] = lo_slice

            interpolated.sort()
            img.ImgAugmentations.interpolated_update(interpolated, extrapolated)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """Checks if interpolation of the file is required

        Args:
            img (ReconVolumeOrFile): ReconstructedVolume or ReconstructedFile

        Returns:
            bool: Boolean if interpolation is needed
        """
        empty = img.ImgAugmentations.empty_slices
        if not empty or len(empty) == 0:
            return False
        return True


# We could also resample by Field of View
class Resample(ImgHandler):
    """Resample image volume

    Raises:
        TypeError: Raised if no-resampling is specified
    """
    def __init__(self, ratio: float = None, dims: list = None,
                 voxel_size: list = None, dz_limit: float = None,
                 dz_goal: float = None, dz_exact: float = None,
                 fill: bool = False, smooth: bool = False,
                 smooth_kernel_size: int = 2):
        """Resamples volumetric image to dimensions, coordinates or voxel size

        Args:
            ratio (float, optional): Resampling ratio, either per-dimension or uniform
                with r>1 upsampling and r<1 downsampling. Defaults to None.
            dims (list, optional): Number of voxels per-dimension to resmaple to, with
                dimensions of None left un-sampled. (e.g. [512, 512, None]). Defaults to None.
            voxel_size (list, optional): Voxel size, in mm, to resample image. Defaults to None.
            dz_limit (float, optional): Limited slice thickness, with dz > dz_limit being
                resampled to dz_goal or 1/2 slice thickness otherwise. Defaults to None.
            dz_goal (float, optional): Resampling goal if dz_limit is specified. Defaults to None.
            dz_exact (float, optional): Resamples all images to this exact dz, if
                specified. Will override dz_limit and dz_goal. Defaults to None.
            fill (bool, optional): Fill holes in resampled RTSTRUCT. Defaults to False.
            smooth (bool, optional): Median value filter smoothing applied to resampled RTSTRUCT.
                Default smooth kernel size is 2. Defaults to False.
            smooth_kernel_size (int, optional): Number of voxels to use in the median filter
                smoothing function. Defaults to 2.

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
        self.fill = fill
        self.smooth = smooth
        self.smooth_kernel_size = smooth_kernel_size


    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """Downsample an image by a specified ratio

        Args:
            img (ReconstructedVolume): Image to resample

        Returns:
            ReconstructedVolume: Downsampled image array
        """
        if self.ratio is not None:
            if type(self.ratio) is int or type(self.ratio) is float:
                current_ratio = [self.ratio for _ in range(3)]
            else:
                current_ratio = self.ratio
        elif self.dims is not None:
            current_ratio = self._dims_to_ratio(img)
        elif self.voxel_size is not None:
            current_ratio = self._voxel_size_to_ratio(img)
        elif self.dz_exact is not None or self.dz_limit is not None:
            current_ratio = [1, 1, 1]  # placeholder for z resampling
        else:
            raise TypeError('Need to specify a rescaling attribute')

        if self.dz_exact is not None or self.dz_limit is not None:
            current_ratio = self._dz_limit_to_ratio(img, previous=current_ratio)

        did_resample = False
        for name, volume in img.volumes.items():
            datatype = volume.dtype
            if not np.all(current_ratio == 1):
                if img.Modality == 'RTSTRUCT':
                    resampled = np.array(zoom(volume, current_ratio, order=0), dtype=datatype)
                    if self.fill:
                        resampled = utils.fill_holes(resampled)
                    if self.smooth:
                        resampled = utils.smooth_median(resampled, self.smooth_kernel_size)
                else:
                    resampled = np.array(zoom(volume, current_ratio), dtype=datatype)

                img.volumes[name] = resampled
                did_resample = True

        if not img.ImgAugmentations.resampled and did_resample:
            img.ImgAugmentations.resampled_update(img.dims.voxel_size, current_ratio)
            img.dims.resampled_update(current_ratio)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        """Determines if the volume or file requires resampling

        Args:
            img (ReconVolumeOrFile): Image volume or file

        Returns:
            bool: Requires resampling

        Notes:
            Resampling hierarchy, where resampling occurs if any is True:
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
        """Computes resampling ratio between two lists

        Args:
            list1 (list): Current parameters
            list2 (list): Desired parameters

        Returns:
            list: Resampling ratio
        """
        ratio = []
        for x0, x1 in zip(list1, list2):
            if x1 is None:
                x1 = x0
            ratio.append(round(x1 / x0, 4))
        return ratio

    def _dims_to_ratio(self, img: ReconVolumeOrFile) -> list:
        """Calculates the resampling ratio based on dimensions
            Where dims * ratio = new size
        """
        return self._compute_ratio(img.dims.shape, self.dims)

    def _voxel_size_to_ratio(self, img: ReconVolumeOrFile) -> list:
        """Calculates the resampling ratio based on voxel size
            Where current size / ratio = new size
        """
        return self._compute_ratio(self.voxel_size, img.dims.voxel_size)

    def _dz_limit_to_ratio(self, img: ReconVolumeOrFile, previous: list) -> list:
        """Calculates the z-axis resampling ratio based on slice thickness limit

        Args:
            img (ReconVolumeOrFile): Image volume or file
            previous: (list): Previous list of resampling ratios

        Returns:
            list: Resampling ratio
        """
        if self.dz_exact is not None:
            if img.dims.dz != self.dz_exact:
                previous[-1] = img.dims.dz / self.dz_exact
        elif img.dims.dz > self.dz_limit:
            if self.dz_goal is not None:
                previous[-1] = img.dims.dz / self.dz_goal
            else:
                previous[-1] = 2
                while (img.dims.dz / previous[-1]) > self.dz_limit:
                    previous[-1] += 2
        return previous


class Crop(ImgHandler):
    def __init__(self, crop_size: list, centroid: list = None, centroids: dict = None,
                 in_patient_coords: bool = False):
        """Cropping function for image and rtstruct volumes

        Args:
            crop_size (list): N length list with int for each volume dimension
            centroid (list, optional): A centroid to crop around, specified
                in voxels. Defaults to None.
            centroids (dict, optional): A dictionary with FrameOfReferenceUID as key
                and the corresponding voxel centroid location as value. Overridden by a specified
                centroid value. Defaults to None.
            in_patient_coords (bool, optional): Specifies if the provided centroids are in
                the patient coordinate system, in mm. Default behavior is image coordinates,
                in voxels. Defaults to False.
        """
        self.crop_size = crop_size
        self.centroid = centroid
        self.centroids = centroids
        self.in_patient_coords = in_patient_coords
        self._centroids = {}

    def _convert_to_voxels(self, centroid_mm: np.ndarray,
                           volfile: ReconstructedVolume) -> np.ndarray:
        """When given a centroid in patient coordinates (mm), convert to voxels

        Args:
            centroid_mm (np.ndarray): Centroid location in patient coordinates
            volfile (ReconstructedVolume): Associated reconstructed volume file

        Returns:
            np.ndarray: Numpy array of integer voxel locations
        """
        centroid_mm_diff = abs(centroid_mm - volfile.dims.origin)
        centroid_vox = centroid_mm_diff / volfile.dims.voxel_size
        return np.array(np.round(centroid_vox), dtype=np.int)

    def _function(self, img: ReconVolumeOrFile) -> ReconVolumeOrFile:
        """Crops an image volume and updates headers accordingly

        Args:
            img (ReconVolumeOrFile): Image volume object to crop

        Returns:
            ReconVolumeOrFile: Cropped volume file

        Notes:
            Centroid will not be observed if the cropped volume will be
            smaller than the crop_size. Will shift centroid to maintain
            the specific crop size
            This function currently does not update the ReconstructedVolume header
            or ReconstructedVolume.dicom_header to reflect the new dimensions
        """
        # Need to update the VolumeDimensions header and dicom header too
        imgshape = np.array(img.dims.shape)
        img_coords = np.zeros((2, len(imgshape)), dtype=np.int)
        patient_coords = np.zeros((2, len(imgshape)), dtype=np.float)

        if self.centroid is not None:
            this_centroid = self.centroid
        elif self.centroids is not None:
            ancestors = img._parent.ancestors
            for ancestor in ancestors:
                if type(ancestor) is FrameOfRef:
                    frame = ancestor
            this_centroid = self.centroids[frame.name]
        else:
            this_centroid = np.array(img.dims.shape) // 2

        if this_centroid is None:
            utils.colorwarn(f'Cannot crop {img.name}, no centroid')
            return img

        if self.in_patient_coords:
            this_centroid = self._convert_to_voxels(this_centroid, img)

        for i, (point, size) in enumerate(zip(this_centroid, self.crop_size)):
            low = max(0, point - size // 2)
            high = low + size
            if high > (imgshape[i] - 1):
                high = imgshape[i] - 1
                low = high - size
                if low < 0:
                    utils.colorwarn(f'Crop size is larger than volume array for {img.PatientID}')
            img_coords[0, i] = low
            img_coords[1, i] = high

        coordrange = img.dims.coordrange()
        for i, (low, high) in enumerate(img_coords.T):
            patient_coords[0, i] = coordrange[i][low]
            patient_coords[1, i] = coordrange[i][high]

        xlo, xhi, ylo, yhi, zlo, zhi = img_coords.T.flatten()

        for name, volume in img.volumes.items():
            datatype = volume.dtype
            img.volumes[name] = np.array(volume[xlo: xhi, ylo: yhi, zlo: zhi], dtype=datatype)

        img.ImgAugmentations.crop_update(img_coords, patient_coords, imgshape)
        img.dims.crop_update(img_coords)
        return img

    def _check_needed(self, img: ReconVolumeOrFile) -> bool:
        passes = [x == y for x, y in zip(self.crop_size, img.dims.shape)]
        return not all(passes)


def calculate_centroids(tree: NodeMixin, method: object, modalities: list = None,
                        structures: list = None, struct_filter: object = None,
                        volume_filter: object = None, nthreads: int = None,
                        offset_fn: object = None) -> dict:
    """Multithreaded computation of the centroid for each frame of reference

    Args:
        tree (NodeMixin): Tree to iterate through
        method (object): Function to calculate centroid. Takes an array, returns
            a list of N items corresponding to the centroid in each axis.
        modalities (list, optional): Structure name to use for centroid. Defaults to None.
        structures (list, optional): List of str to calculate the centroid. Defaults to None.
        struct_filter (object, optional): A custom filtering function to pick structures, this is
            overridden if the paremeter structures is also specified. Function should accept a
            structure name and return a boolean for calculation of centroid. Defaults to None.
        volume_filter (object, optional): A custom filtering function to pick volumes, this is
            overridden if the parameter structures is also specified. Function should accept a
            ReconstructedVolume object and return a boolean for calculation of
            centroid. Defaults to None.
        nthreads (int, optional): Number of threads to use for the computation. Higher
            threads may run faster with higher memroy usage. Defaults to CPU count // 2.
        offset_fn (object, optional): A function to offset the centroid. Should accept a the
            centroid and a ReconstructedVolume object and return a centroid of
            equivalent dimensions. Defaults to None.

    Returns:
        dict: Keyed to Frame Of Reference UID with centroid voxel location as value
    """
    def name_check(structures: list) -> object:
        def fn(name):
            if structures is None:
                return True
            if type(structures) is dict:
                for key, values in structures.items():
                    if name == key or name in values:
                        return True
            return name in structures
        return fn

    def only_modality(modalities: list) -> object:
        def fn(volfile):
            return volfile.header['Modality'] in modalities
        return fn

    if modalities is not None:
        if type(modalities) is str:
            modalities = [modalities]
        volume_filter = only_modality(modalities)
    elif modalities is None and structures is not None:
        modalities = ['RTSTRUCT']

    if structures is not None:
        if type(structures) is str:
            structures = [structures]
        struct_filter = name_check(structures)

    def method_wrapper(method, offset_fn=None):
        def fn(frame):
            for volfile in frame.iter_volumes():
                if volume_filter is None or volume_filter(volfile):
                    original_fmt = type(volfile)
                    if original_fmt is ReconstructedFile:
                        volfile.load_array()

                    for name, volume in volfile.volumes.items():
                        if struct_filter is None or struct_filter(name):
                            CoM = np.array(np.round(method(volume)), dtype=np.int)
                            if offset_fn is not None:
                                CoM = offset_fn(CoM, volfile)
                            if original_fmt is ReconstructedFile:
                                volfile.convert_to_file()
                            return (frame.name, CoM)

                    if original_fmt is ReconstructedFile:
                        volfile.convert_to_file()
            return (frame.name, None)
        return fn

    if method is None:
        raise TypeError('Must specify method to compute centroids')
    else:
        if not nthreads:
            nthreads = multiprocessing.cpu_count() // 2

        centroids = {}
        compute_fn = method_wrapper(method, offset_fn)
        with ThreadPool(max_workers=nthreads) as P:
            points = list(P.map(compute_fn, tree.iter_frames()))
            P.shutdown()

        for name, centroid in points:
            if centroid is None:
                utils.colorwarn(f'Frame Of Reference {name} has no centroid')
            centroids.update({name: centroid})

    return centroids


"""
from scipy.interpolate import interpn
from scipy.ndimage.morphology import distance_transform_edt

Start of Code for the interpolation of RTSTRUCTs. These will require more user specification
to prevent the interpolation of intentional gaps within the structure sets.

@handle_files
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
