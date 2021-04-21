from groupings import Cohort
from glob import glob
import numpy as np
import utils
import time
import os
import psutil
'''
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''
utils.clear_runtime()

files = glob('/home/eporter/eporter_data/hippo_data/4*/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=False)

start = time.time()
cohort.recon(in_memory=False)
process = psutil.Process(os.getpid())
print(cohort)
print('All:', process.memory_info().rss * 10e-9)
print('elapsed time:', time.time()-start)
utils.average_runtime()

"""
cohort.volumes_to_pointers()
process = psutil.Process(os.getpid())
print(cohort)
print('Pointers:', process.memory_info().rss * 10e-9)

cohort.pointers_to_volumes()
process = psutil.Process(os.getpid())
print(cohort)
print('Loaded:', process.memory_info().rss * 10e-9)
#cohort.save_tree('/home/eporter/eporter_data/')

for patient in cohort:
    for study in patient:
        for ref in study:
            vol = ref.recon()
            try:
                rts = vol.struct[0].volumes['hippocampus_l_ep']
                utils.three_axis_plot(vol.ct[0], 'ct0', rts)
                utils.three_axis_plot(vol.mr[0], 'mr0', rts)
                print('done plotting')
            except Exception:
                pass
            temp = np.zeros((1, *vol.ct[0].shape))
            temp[0, 100:150, 100:150, 50:75] = 1
            temp[0, 110:120, 110:120, 51:74] = 0
            print(temp.shape)
            print(ref)
            print('here')
            try:
                ref.decon.from_ct(temp)
            except Exception:
                pass
            else:
                print(ref)
            """



'''
TODO:
- Check (ensure works properly)
    - associated multi-leaf functionality
    - save volumes and dicoms functions
- Integrate (adapt to current code)
    - tools for modifications
- Design
    - Generators (pytorch and tensorflow)
    - Nifti conversion and saving
    - Loading saved volumes
    - Multithreaded save on reconstruction
'''
