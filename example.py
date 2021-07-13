import time
import sys
from glob import glob
from . import tools
from . import utils
from .groupings import Cohort, ReconstructedVolume
import numpy as np
from scipy.ndimage import center_of_mass


"""
Overview
---------
A full image processing pipeline to take the images form raw
DICOM files to a series of read-to-train numpy arrays which
can be given to a data generator. (Note: future work will be
able to yield a data generator function at the end of this example)

Steps
---------
The purpose of this example is to load DICOM files from MIMExport/,
sort them into a tree and save them to dicoms/.

Then to remove any incomplete studies (those without CT and RTSTRUCT)
and to reconstruct only the complete leaves.

Then the reconstructed volumes are interpolated, resampled to 512x512
in the axial dimensions and a maximum of 2.39mm slice thickness.

A custom centroid is then computed using the functions defined below.

From that centroid the images are normalized and then cropped to 200x200x35
voxels. The tool application function changes the already saved volumes,
therefore saving is not required again.
"""


# Custom functions for center of mass calculation
# User can implement these themselves or use scipy.ndimage.center_of_mass
# It entirely depends on their specific worfklow
def surface_centroid(img: np.ndarray) -> list:
    """[Computes a centroid based on the bone HU range. XY-axis is center of mass
        while the Z-axis is the top of the skull]

    Args:
        img ([type]): [A 3D-CT encoded as a 32-bit numpy array of HU values]

    Returns:
        list: [The center of mass values]
    """
    losurface = img > 650  # HU > 500 to get bone and metal
    hisurface = img < 2000  # HU < 3000 to exclude metal
    only_bones = losurface * hisurface
    surface = utils.clean_up(only_bones)
    xy_projection = np.sum(surface, axis=-1)
    xy_com = np.array(np.round(center_of_mass(xy_projection)), dtype=np.int)
    z_projection = np.sum(surface, axis=(0, 1)) > 100
    z_top = np.nonzero(z_projection)[0].min()
    return np.array([xy_com[0], xy_com[1], z_top])


def offset_centroid(offset: list) -> object:
    """[A function used to offset the center of mass by a specified distance,
        given in milimeters]

    Args:
        offset (list): [The offset center of mass, in mm]

    Returns:
        object: [A function which takes a center of mass and ReconstructedVolume
            type object and returns a list of equal dimensions offset by a given
            offset distance.]
    """
    def fn(CoM: list, volfile: ReconstructedVolume) -> list:
        dist_mm = np.array(offset)  # in mm
        dist_vox = np.array(np.round(dist_mm / volfile.dims.voxel_size), dtype=np.int)
        return CoM + dist_vox
    return fn


if __name__ == '__main__':
    start = time.time()
    # Glob all unsorted files
    files = glob('/home/eporter/eporter_data/rtog_project/MIMExport/**/*.dcm', recursive=True)

    # Sort files into tree of type Cohort
    cohort = Cohort(name='RTOG_Hippocampus', files=files, include_series=False)
    print(cohort)

    # Save sorted dicom files
    cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')

    # Filter to remove and MR from the tree
    modalities = {'Modality': ['CT', 'RTSTRUCT']}   # Without exact
    excluded = cohort.pull_incompletes(group='Study', exact=False,
                                       contains=modalities, cleaned=True)
    print(cohort)

    # Reconstruct dicoms into arrays onto disk at the specified path
    filter_rt = {'StructName': {'hippocampus': ['hippocampus'],
                                'hippo_avoid': ['hippoavoid', 'hippo_avoid']}}
    cohort.recon(parallelize=True, in_memory=False,
                 path='/home/eporter/eporter_data/rtog_project/built/', filter_by=filter_rt)
    print(cohort)

    # Apply interpolation, resampling
    toolset = [tools.Interpolate(extrapolate=True),
               tools.Resample(dims=[512, 512, None], dz_limit=2.39)]
    cohort.apply_tools(toolset)

    # Calculate the centroids based on center of mass of hippocampus
    centroids = tools.calculate_centroids(tree=cohort, modalities=['CT'], method=surface_centroid,
                                          offset_fn=offset_centroid([3.61, -1.04, 93.2]))

    # Normalize the images and then crop the volumes
    toolset = [tools.Normalize(),
               tools.Crop(crop_size=[200, 200, 35], centroids=centroids)]
    cohort.apply_tools(toolset)

    # Report elapsed computational time
    print('elapsed:', time.time() - start)
    sys.stdout.flush()
