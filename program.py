from groupings import Cohort
from glob import glob

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

# cohort.save_tree('/home/eporter/eporter_data/', prefix='date')
