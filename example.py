from groupings import Cohort
from glob import glob
import time
import sys
import tools
import utils
import anytree
#from pathos.pools import ProcessPool
from concurrent.futures import ProcessPoolExecutor as ProcessPool
from concurrent.futures import ThreadPoolExecutor as ThreadPool

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
files = glob('/home/eporter/eporter_data/rtog_project/MIMExport/**/*.dcm', recursive=True)
# Sort files into tree
cohort = Cohort(name='RTOG_Hippocampus', files=files, include_series=False, filter_by=filter_list)
print(cohort)

# Save sorted dicom files
cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')

# Reconstruct dicoms into arrays at specified path
print('tree saved, now reconstructing')
cohort.recon(parallelize=True, in_memory=False, path='/home/eporter/eporter_data/rtog_project/built/')

# Apply interpolation function
print('applying tools')
#toolset = [tools.Interpolate()]
toolset = [tools.Interpolate(extrapolate=True)]
cohort.apply_tools(toolset)
print(cohort)

for vol in cohort.iter_volumes():
    for name, files in vol.items():
        f = files[0]
        print(f.ImgAugmentations.interpolated)


print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()