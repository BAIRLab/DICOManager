import time
import sys
from glob import glob
from . import tools
from . import utils
from .groupings import Cohort
import numpy as np
from scipy.ndimage import center_of_mass
import optparse


# Custom functions for center of mass calculation
def surface_centroid(img):
    losurface = img > 650  # HU > 500 to get bone and metal
    hisurface = img < 2000  # HU < 3000 to exclude metal
    only_bones = losurface * hisurface
    surface = utils.clean_up(only_bones)
    xy_projection = np.sum(surface, axis=-1)
    xy_com = np.array(np.round(center_of_mass(xy_projection)), dtype=np.int)
    z_projection = np.sum(surface, axis=(0, 1)) > 100
    z_top = np.nonzero(z_projection)[0].min()
    return np.array([xy_com[0], xy_com[1], z_top])


def offset_centroid(offset):
    def fn(CoM, volfile):
        dist_mm = np.array(offset)  # in mm
        dist_vox = np.array(np.round(dist_mm / volfile.dims.voxel_size), dtype=np.int)
        return CoM + dist_vox
    return fn


def calc_z(x_dim, y_dim):
    return int(round((200*200*35)/(x_dim*y_dim)))


usage = "example -x # -y #"
parser = optparse.OptionParser(usage)

parser.add_option('-x', '--xaxis', action='store', dest='x_dim', help='x voxel size', default=200)
parser.add_option('-y', '--yaxis', action='store', dest='y_dim', help='y voxel size', default=200)
options, args = parser.parse_args()

X_DIM = int(options.x_dim)
Y_DIM = int(options.y_dim)
Z_DIM = calc_z(X_DIM, Y_DIM)

'''
# in the format of:
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''
# StructName of dict will rename structures in the value to the key
filter_list = {'StructName': {'hippocampus': ['hippocampus'],
                              'hippo_avoid': ['hippoavoid', 'hippo_avoid']},
               'Modality': ['CT', 'RTSTRUCT']}

start = time.time()
# Glob all unsorted files
files = glob('/home/eporter/eporter_data/rtog_project/dicoms/*/Patient_0933-*/**/*.dcm', recursive=True)

# Sort files into tree
cohort = Cohort(name='RTOG_Hippocampus', files=files, include_series=False, filter_by=filter_list)
print(cohort)

# Filter test 1
#filter_list = {'Modality': ['CT', 'RTSTRUCT']}   # Without exact
#excluded = cohort.pull_incompletes(group='FrameOfRef', exact=False, contains=filter_list)  # rename to contains

# Filter test 2
#filter_list = {'Modality': ['CT', 'RTSTRUCT', 'MR', 'MR']}  # With exact
#excluded = cohort.pull_incompletes(group='Patient', exact=True, contains=filter_list)  # rename to contains

# Save sorted dicom files
#cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')

# Reconstruct dicoms into arrays at specified path
cohort.recon(parallelize=True, in_memory=False, path='/home/eporter/eporter_data/rtog_project/built/')
#print(cohort)

# Apply interpolation, resampling
toolset = [tools.Interpolate(extrapolate=True),
           tools.Resample(dims=[512, 512, None], dz_limit=2.39)]

cohort.apply_tools(toolset)

# Calculate the centroids based on center of mass of hippocampus
centroids = tools.calculate_centroids(tree=cohort, modalities=['CT'], method=surface_centroid,
                                      offset_fn=offset_centroid([3.61, -1.04, 93.2]))

pre_crop = utils.structure_voxel_count(cohort, 'hippocampus')

toolset = [tools.Normalize(),
           tools.Crop(crop_size=[X_DIM, Y_DIM, Z_DIM], centroids=centroids)]

# Need to:
# - remove extra functions above
# - integrate those into tools.py
# - add in parser
# - print the coverage and the parameters at end of file
# - cue up a bunch of them with task spooler
# - do a [100:220:5] grid search with z floating to the nearest, smaller int
#   using z = int(round((200*200*35)/(x*y)))
# - automate the queing process using a script

cohort.apply_tools(toolset)
post_crop = utils.structure_voxel_count(cohort, 'hippocampus')

bulks = []
for name, presum in pre_crop.items():
    postsum = post_crop[name]
    ratio = postsum / presum
    bulks.append(ratio)

print(f'Dimensions: [{X_DIM}, {Y_DIM}, {Z_DIM}]')
print(f'Included hippocampus: {np.mean(bulks)} +/- {np.std(bulks)}')
print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()
