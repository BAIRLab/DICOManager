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

files = glob('/home/eporter/eporter_data/hippo_data/**/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=False)

start1 = time.time()
cohort = utils.threaded_recon(cohort)
print('elapsed1:', time.time() - start1)
print(cohort)

start0 = time.time()
cohort.recon(in_memory=False)
print('elapsed0:', time.time() - start0)
