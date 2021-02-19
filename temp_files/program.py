from groups import Cohort

path = '/data/imported_data/'
#path = '/home/eporter/eporter_data/hippo_data'

group = Cohort(path)

print(group)

for patient in group:
    for study in patient:
        print(study)