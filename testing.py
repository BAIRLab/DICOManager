from pydicom.data import get_testdata_files
import difflib
import numpy as np
import os
import pydicom
import glob
import time
import reconstruction as recon
import deconstruction as decon

patients = sorted(os.listdir('/home/eporter/eporter_data/optic_structures/dicoms/'))
rt_series = glob.glob('/home/eporter/eporter_data/optic_structures/dicoms/' + patients[0] + '/RTSTRUCT/*.dcm')
ct_series = glob.glob('/home/eporter/eporter_data/optic_structures/dicoms/' + patients[0] + '/CT/*.dcm')
built = recon.struct(rt_series[0], wanted_contours=['skull'])
test = decon.from_ct(ct_series, built, roi_names=['testing'])
name = test.SOPInstanceUID + '.dcm' 
decon.save_rt(test, name)
print(f'DICOM saved to: {os.getcwd()}/{name}')