from groupings import Cohort
from glob import glob
import numpy as np
import utils
import time
import os
import psutil
from concurrent.futures import ProcessPoolExecutor as ProcessPool
'''
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''
utils.clear_runtime()

files = glob('/home/eporter/eporter_data/hippo_data/**/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=False)

print('Reconstructing')
start = time.time()

temp0 = Cohort(name='Temp0', files=None, include_series=False)
temp1 = Cohort(name='Temp1', files=None, include_series=False)
temp2 = Cohort(name='Temp2', files=None, include_series=False)

count = 0
for patient in cohort:
    count += 1
    temp0.adopt(patient)
    if count > 10:
        break

count = 0
for patient in cohort:
    count += 1
    temp1.adopt(patient)
    if count > 10:
        break

count = 0
for patient in cohort:
    count += 1
    temp2.adopt(patient)
    if count > 10:
        break

def fn(obj):
    return obj.recon(in_memory=False)

with ProcessPool() as P:
    trees = list(P.map(fn, [cohort, temp0, temp1, temp2]))
"""
cohort.recon(in_memory=False)
temp0.recon(in_memory=False)
temp1.recon(in_memory=False)
temp2.recon(in_memory=False)
"""
print(trees)
print('elapsed time:', time.time()-start)
#utils.average_runtime()

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
