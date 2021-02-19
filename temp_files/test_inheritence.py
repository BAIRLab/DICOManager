from dataclasses import dataclass, field
from typing import TypeVar, Generic

class A:
    def __init__(self, value=10):
        self.value = value

    def add_five(self):
        self.value += 5

    def print_value(self):
        print(self.value)

@dataclass
class B:
    one: int
    two: int
    three: int

    def pval(self, name):
        print(self.__dataclass_fields__['name'])

    def __getitem__(self, name):
        return getattr(self, name)

b = B(1, 2, 3)
print(b['one'])

test = {}
if 't' not in test:
    test['t'] = 1

print(test)
new = test.pop('t')

def split_to_dict(self, ObjectType, data_field):
    # we would need to figure out how to specify an object type
    # how to specify what the main data point is for the lower tier
    fill_dict = {}
    while self.filelist:
        f = self.filelist.pop()
        if f[data_field] not in fill_dict:
            fill_dict[f[data_field]] = f
        else:
            fill_dict[f[data_field]].append(f)

    for key, data in fill_dict.items():
        self.data[key] = ObjectType(data)

# Called in cohort:
# split_to_dict(Patient, 'PatientID')

# Called in patient:
# split_to_dict(Study, 'StudyID')

# Called in study:
# split_to_dict(Series, 'SeriesID')

# In Series:
# split_to_dict(FileList, 'Modality)


import dicomanager as dm

cohort = dm.cohort('/path/', filter=list_of_mrns)
cohort2 = dm.cohort('/path/', filter=second_list)

for patient in cohort:
    for study in patient:
        for series in study:
            ct_group = series.ct  # List of files for CT
            ct_vol = series.recon.ct()  # Reconstructed CT numpy array
            ct_vol = series.recon.ct(type='whatever')  # Reconstructed CT whatever type
            struct = series.recon.struct()
            ct_wl = dm.tools.window_level(ct_vol, window=1, level=2)

cohort.save('/path/')  # Writes a DICOM tree as configured

volumes = cohort.reconstruct(save=True)  # Reconstruct tree with volumes at the leaves
volumes.save('/path/')  # Write a array tree as configured

patient = cohort['patientID']
study2 = patient.study[2]
patient.study[0].merge(patient.study[1])  # Of equal level list = list1 + list2
patient.study[0].append(study2.series[1])  # Of lower level list.append(item)
patient.remove(patient.study[2])  # More useful in an interactive environment
all_files = patient.all_files()  # Returns a list of all files for that patient

print(patient)
"""
Patient:MRN_PatientName
    Study:Date_Time_Description
        Series:Date_Time_Description
            CT0:105
            CT1:105
            RTSTRUCT:2
            RTDOSE:1
        Series:Date_Time_[None]
            CT0:225
            MR0:200
"""

"""
Patient
    ._frame_of_references([FileGroup0, FileGroup1])
    Study0
        Series
            SimCt .FileGroup
    Study1
        Series
            CBCT0
    Study2
        Series
            CBCT1
    Study3 CBCT2
    Study4 CBCT3
    Study5 CBCT4
"""