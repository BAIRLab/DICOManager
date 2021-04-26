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

start = time.time()
files = glob('/home/eporter/eporter_data/hippo_data_fldr/hippo_data/1*/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=True)
#cohort = utils.threaded_recon(cohort)
cohort.recon(parallelize=True)
print(cohort)
print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()
