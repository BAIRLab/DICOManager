from groupings import Cohort
from tqdm import tqdm
from glob import glob

files = glob('/home/eporter/eporter_data/hippo_data/4*/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=True)
#print(cohort)

for patient in cohort:
    for study in patient:
        for ref in study:
            vol = ref.recon()
            print(vol)
    #print(patient.datename)

#cohort.save_tree('/home/eporter/eporter_data/', prefix='date')

# Read from a json
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}

# Pass filter list into cohort, keep passing it downward
