from groupings import Cohort
from glob import glob
import numpy as np

'''
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''

files = glob('/home/eporter/eporter_data/hippo_data/4*/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=True)

for patient in cohort:
    for study in patient:
        for ref in study:
            vol = ref.recon()
            print(vol)
            temp = np.zeros((1, *vol.ct[0].shape))
            temp[0, 100:150, 100:150, 50:75] = 1
            temp[0, 110:120, 110:120, 51:74] = 0
            print(temp.shape)
            print(ref)
            print('here')
            ref.decon.from_ct(temp)
            print(ref)
            raise TypeError


# cohort.save_tree('/home/eporter/eporter_data/', prefix='date')
