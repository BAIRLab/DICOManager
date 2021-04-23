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

#files = glob('/home/eporter/eporter_data/hippo_data/**/**/*.dcm', recursive=True)
start2 = time.time()
files = glob('/home/eporter/eporter_data/hippo_data_fldr/hippo_data/**/**/*.dcm', recursive=True)
print(len(files))
cohort = Cohort(name='TestFileSave', files=files, include_series=False)
print('build tree time:', time.time() - start2)

start1 = time.time()
cohort = utils.threaded_recon(cohort)
print(cohort)
print('elapsed1:', time.time() - start1)

start0 = time.time()
cohort.recon(in_memory=False)
print('elapsed0:', time.time() - start0)
