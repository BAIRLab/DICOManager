#!/usr/bin/python

import pandas as pd
import pydicom
import os
import shutil
import glob
import struct
import csv
import optparse
from tqdm import tqdm

__author__ = "Evan Porter, Ron Levitin, Nick Myziuk"
__copyright__ = "Copyright 2018, Beaumont Health"
__credits__ = ["Evan Porter", "Nick Myziuk", "Thomas Guerrero"]
__maintainer__ = "Evan Porter"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research & Data Organization"

# Alter this path for directories not stored with '/data'
parser = optparse.OptionParser()

parser.add_option('-d', '--date', action='store_true', dest='date',
                  help='sort modalities by date', default=False)
parser.add_option('-f', '--file', action='store', dest='csv_file',
                  help='MRN csv to sort from', default=None)
parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Base storing directory', default='pwd')
parser.add_option('-l', '--legacy', action='store',
                  dest='legacy', help='legacy dir', default=None)

options, args = parser.parse_args()

if not options.csv_file:
    raise NameError('A sorting csv file must be specified with flag --file')

if options.base_dir == 'pwd':
    options.base_dir = os.getcwd()
elif options.base_dir[-1] == '/':
    options.base_dir = options.base_dir[:-1]

if options.legacy:
    if options.legacy[-1] == '/':
        options.legacy = options.legacy[:-1]
    legacy_files = glob.glob(options.legacy + '/**/*.dcm', recursive=True)
    for lf in legacy_files:
        shutil.copy(lf, options.base_dir + '/imported_data/')


def write_to_path(file_path, patientID, dicom_file, data_dir, date=None,
                  subfolder=False, **kwargs):
    """
    Function
    ---------
    Write the dicom_file into a folder at file_path

    Parameters
    ---------
    file_path, data_dir : str
        Strings to the paths of the data directory and the final file
            path locale
    patientID : int
        The patient's Medical Record Number (MRN) identifier
    dicom_file, data_dir : str -> .dcm file
        A dicom file to be move to file_path
    subfolder : bool (Default = False)
        A subfolder to be placed within the file_path
    """
    if options.date and date:
        new_path = os.path.join(file_path, patientID, date, subfolder)
    elif subfolder:
        new_path = os.path.join(file_path, patientID, subfolder)
    else:
        new_path = os.path.join(file_path, patientID)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    _, filename = os.path.split(dicom_file)
    destination = os.path.join(new_path, filename)
    shutil.move(os.path.join(data_dir, dicom_file), destination)


def specific_sort(dicom_file, file_path, cohort_list,
                  patientID, modality, ds):
    """
    Function
    ----------
    To sort the provided files into their correct paths

    Parameters
    ---------
    dicom_file, file_path : str
        A path to the file and a path to where the file will be moved
    MRN_list : list
        A list of MRN's corresponding to files which should be moved
    patientID : int
        An integer of the patient ID, also known as the MRN
    description : str
        A description of the study from the dicom header
    modality : str
        A string representing the modality
    """
    # List is append, subfolder if we add more modalities, we just
    #   update this dictionary to include the options
    with open(options.base_dir + '/' + options.csv_file, mode='r') as csv_file:
        input_file = csv.reader(csv_file)
        subfolders = dict((rows[0], rows[1]) for rows in input_file)

    write_params = {"file_path": file_path,
                    "patientID": patientID,
                    "dicom_file": dicom_file,
                    "data_dir": data_dir,
                    "date": None
                    }

    if hasattr(ds, 'AcquisitionDate'):
        write_params['date'] = ds.AcquisitionDate
    elif hasattr(ds, 'StudyDate'):
        write_params['date'] = ds.StudyDate

    try:
        if int(patientID) in cohort_list:
            if modality in subfolders.keys():
                subfolder = subfolders[modality]
            else:
                subfolder = ds.StudyDescription
    except ValueError:
        if patientID in str(cohort_list):
            if modality in subfolders.keys():
                subfolder = subfolders[modality]
            else:
                subfolder = ds.StudyDescription
    finally:
        write_params.update({"subfolder": subfolder})
        write_to_path(**write_params)


data_dir = options.base_dir + '/imported_data'
dicom_files = glob.glob(data_dir + '/*.dcm*')
csv_path = options.base_dir + '/sort_csv/' + options.csv_file

try:
    cohort_list = pd.read_csv(csv_path, engine='python')['MRN'].tolist()
except FileNotFoundError:
    raise FileNotFoundError(f'The specified .csv is not in {csv_path}')
else:
    for _, dicom_file in enumerate(tqdm(dicom_files)):
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, struct.error):
            shutil.move(dicom_file, options.base_dir + '/rejected_files')
        else:
            if "StudyDescription" not in dir(ds):
                ds.add_new([0x0008, 0x1030], 'LO', '')
            if "Modality" in dir(ds):
                project_dir = csv_path[:-4].rpartiont('/')[-1]
                if ds.StudyDescription[:4] == 'CBCT':
                    specific_sort(dicom_file=dicom_file,
                                  file_path=project_dir,
                                  cohort_list=cohort_list,
                                  patientID=ds.PatientID,
                                  modality='CBCT',
                                  ds=ds)
                else:
                    specific_sort(dicom_file=dicom_file,
                                  file_path=project_dir,
                                  cohort_list=cohort_list,
                                  patientID=ds.PatientID,
                                  modality=ds.Modality,
                                  ds=ds)
