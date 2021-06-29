import time
import sys
import tools
from groupings import Cohort
from glob import glob
import numpy as np

'''
# in the format of:
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''
filter_list = {'StructName': {'hippocampus': ['hippocampus'],
                              'hippo_avoid': ['hippoavoid', 'hippo_avoid']},
               'Modality': ['CT', 'RTSTRUCT']}

start = time.time()

# Glob all unsorted files
files = glob('/home/eporter/eporter_data/rtog_project/MIMExport/**/*.dcm', recursive=True)

# Sort files into tree
cohort = Cohort(name='RTOG_Hippocampus', files=files, include_series=False, filter_by=filter_list)
#print(cohort)

# Save sorted dicom files
cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')

# Reconstruct dicoms into arrays at specified path
cohort.recon(parallelize=True, in_memory=False, path='/home/eporter/eporter_data/rtog_project/built/')

# Apply interpolation function
# Working: extrapolate, normalize, standardize, window level, resampling
# Untested: BiasFieldCorrection, cropping

toolset = [tools.Interpolate(extrapolate=True), tools.Resample(dims=[512, 512, None], dz_limit=2.39)]
cohort.apply_tools(toolset)
#print(cohort)

fovs = []
vox_sizes = []

for vol in cohort.iter_volumes():
    for name, files in vol.items():
        f = files[0]
        print(f.PatientID, f.dims.shape)
        fovs.append(np.round(f.dims.field_of_view, 2))
        vox_sizes.append(np.round(f.dims.voxel_size, 2))

fovs = np.unique(fovs, axis=0)
vox_sizes = np.unique(vox_sizes, axis=0)

print(fovs)
print(vox_sizes)

print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()