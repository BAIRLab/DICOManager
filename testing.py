import os
import glob
import reconstruction as recon
import deconstruction as decon

dir_path = '/home/eporter/eporter_data/optic_structures/dicoms/'
patients = sorted(os.listdir(dir_path))
rt_series = glob.glob(dir_path + patients[0] + '/RTSTRUCT/2*.dcm')
ct_series = glob.glob(dir_path + patients[0] + '/CT/*.dcm')
built = recon.struct(rt_series[0], wanted_contours=['test2', 'test3', 'test4'])
test = decon.to_rt(ct_series, rt_series[0], built, mim=False)
name = test.SOPInstanceUID + '.dcm'
decon.save_rt(test, name)
print(f'DICOM saved to: {os.getcwd()}/{name}')
