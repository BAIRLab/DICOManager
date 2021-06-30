import time
import sys
import tools
from groupings import Cohort
from glob import glob

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
print(cohort)

# Save sorted dicom files
cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')

# Reconstruct dicoms into arrays at specified path
cohort.recon(parallelize=True, in_memory=False, path='/home/eporter/eporter_data/rtog_project/built/')

# Apply interpolation function
# Working: extrapolate, normalize, standardize, window level, resampling, cropping
# Untested: BiasFieldCorrection

centroids = tools.compute_centroids(tree=cohort, structure='hippo_avoid', method='center_of_mass')
# Can also have custom method, will ignore structure parameter
# Custom method is given a frame of reference, returns a list of the centroid voxel indicies as integers

toolset = [tools.Interpolate(extrapolate=True),
           tools.Resample(dims=[512, 512, None], dz_limit=2.39),
           tools.Crop(crop_size=[100, 100, 50], centroids=centroids)]

cohort.apply_tools(toolset)

for f in cohort.iter_modalities():
    print(type(f))
    for v in f.volumes_data.values():
        v = v[0]
        v.load_array()
        for name, volume in v.volumes.items():
            print(name, ': ', volume.shape)

print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()
