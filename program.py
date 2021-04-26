from groupings import Cohort
from glob import glob
import utils
import time
import sys

'''
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''

start1 = time.time()
files = glob('/home/eporter/eporter_data/hippo_data_fldr/hippo_data/**/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=True)
cohort = utils.threaded_recon(cohort)
print(cohort)
print('elapsed1:', time.time() - start1)
print(len(cohort))
sys.stdout.flush()
