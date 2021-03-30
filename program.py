from groupings import Cohort
from tqdm import tqdm
from glob import glob

files = glob('/home/eporter/eporter_data/hippo_data/4*/**/*.dcm', recursive=True)
cohort = Cohort(name='TestFileSave', files=files, include_series=True)
print(cohort)

for patient in cohort:
    for study in patient:
        for ref in study:
            test = ref.recon()
            print(done)
    #print(patient.datename)

#cohort.save_tree('/home/eporter/eporter_data/', prefix='date')
