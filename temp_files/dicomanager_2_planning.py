from dataclasses import dataclass
from dataclasses import fields
from copy import deepcopy
import pydicom
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

"""
Data Storage Classes:
FileData -> FileList -> Series -> Study -> Patient -> Cohort

FileUtils dictates the basic nature of the data storage classes
    Sort is an inner class to FileUtils

Reconstruction, Decontruction, Tools apply to Series
    Those functions apply to:
    ImageVolume -> SeriesVolumeSet

Reconstruction can occur at Study, Patient, Cohort but that is
    a multithreaded call of reconstruction at the series level
    It would be cohort.reconstruct():
        P.map(reconstruct, patients) calling...
        P.map(reconstruct, studies) calling...
        map(reconstruct, series) which is single threaded
    Or, we could build a complete list of series:
        P.map(reconstruct, allseries)
        It would return a 'location' for it to be put
        back into the tree structure.

Project Layout:
StorageClasses.py
    - FileUtils: Base group functionality
    - Cohort
    - Patient
    - Study
    - Series: Modality split data set
    - FileList: List of FileData
    - FileData: Digested DICOMs and POSIX path
Constructors.py
    - Reconstruction
    - Deconstruction
    - VolumeSet : A tree of Series Volume Sets
    - SeriesVolumeSet : A single Series of multiple modalities
    - ImageVolume : A single volume with metadata
Tools.py
    - Tools: Image Processing and User Functions
        - Window/Level, interpret, crop, flip, etc.
    - ImageUtils: Re/Deconstruction Inner Function



I'm designing the UI to be as follows:

import dicomanager as dm

cohort = dm.cohort('/path/', filter=list_of_mrns)
cohort2 = dm.cohort('/path/', filter=second_list)

for patient in cohort:
    for study in patient:
        for series in study:
            ct_group = series.ct  # List of files for CT
            ct_vol = series.recon.ct()  # Reconstructed CT numpy array
            # Reconstructed CT whatever type
            ct_vol = series.recon.ct(type='whatever')
            struct = series.recon.struct()
            ct_wl = dm.tools.window_level(ct_vol, window=1, level=2)

cohort.save('/path/')  # Writes a DICOM tree as configured

# Reconstruct tree with volumes at the leaves
volumes = cohort.reconstruct(save=True)
volumes.save('/path/')  # Write a array tree as configured

patient = cohort['patientID']
study2 = patient.study[2]
patient.study[0].merge(patient.study[1])  # Of equal level list = list1 + list2
patient.study[0].append(study2.series[1])  # Of lower level list.append(item)
patient.remove(patient.study[2])  # More useful in an interactive environment
all_files = patient.all_files()  # Returns a list of all files for that patient

print(patient)

Patient: MRN_PatientName
   Study: Date_Time_Description
       Series: Date_Time_Description
           CT0: 105
            CT1: 105
            RTSTRUCT: 2
            RTDOSE: 1
        Series: Date_Time_[None]
           CT0: 225
            MR0: 200

"""


# --------------- Inner functions to Series ---------------


