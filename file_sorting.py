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


__author__ = ["Evan Porter", "Ron Levitin", "Nick Myziuk"]
__liscense__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


def _write_to_path(dicom_file, dest_dir, patientID, subfolder=False):
    """
    Function
    ---------
    Write the dicom_file into a folder at file_path

    Parameters
    ---------
    dicom_file, dest_dir : str
        Strings for the current location of the DICOM and its new location
    patientID : int
        The patient's Medical Record Number (MRN) identifier
    subfolder : bool (Default = False)
        A subfolder to be placed within the file_path
    """
    path_list = [dest_dir, patientID]
    
    if subfolder:
        path_list.append(subfolder)

        new_path = os.path.join(*path_list)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    _, filename = os.path.split(dicom_file)
    destination = os.path.join(new_path, filename)
    
    if options.move_file:
        shutil.move(dicom_file, destination)
    else:
        shutil.copy(dicom_file, destination)


def _specific_sort(dicom_file, dest_dir, cohort_list, ds):
    """
    Function
    ----------
    To sort the provided files into their correct paths

    Parameters
    ---------
    dicom_file, dest_dir : str
        Strings for the current location of the DICOM and its new location
    cohort_list : list
        A list of MRN's corresponding to files which should be moved
    ds : pydicom.dataset.FileDataset
        The pydicom dataset header for the DICOM file to be moved
    """

    write_params = {"dest_dir": dest_dir,
                    "patientID": ds.PatientID,
                    "dicom_file": dicom_file,
                    }

    try:
        if int(ds.PatientID) in cohort_list or ds.PatientID in str(cohort_list):
            if "CBCT" in ds.StudyDescription:
                subfolder = "CBCT"
            else:
                subfolder = ds.Modality
            write_params.update({"subfolder": subfolder})
            _write_to_path(**write_params)
    except ValueError:
        pass


# Command line starts below here
parser = optparse.OptionParser()

parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory containing unsorted files', default='/data/imported_data/')
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN csv to sort from, should be located in -b', default=None)
parser.add_option('-m', '--move', action='store_true', dest='move_file',
                  help='Move instead of default of copy', default=False)
parser.add_option('-p', '--project-dest', action='store', dest='project_dir',
                    default = '/data/sorted_data/')

options, args = parser.parse_args()

if not options.csv_file:
    raise NameError('A sorting csv file must be specified with flag --csv')

if options.move_file:
    bad_input = True
    while bad_input:
        confirm = input("Confirm you want to move and not copy the files (y/n): ")
        if confirm.lower() == "n":
            print("Copying instead...")
            bad_input = False
            options.move_file = False
            break
        if confirm.lower() == "y":
            bad_input = False
            break
        else:
            bad_input = True

dicom_files = glob.glob(os.path.join(options.base_dir, '**/*.dcm*'), recursive=True)

try:
    csv_path = os.path.join(options.csv_file)
    cohort_list = pd.read_csv(csv_path, engine='python')['MRN'].tolist()
except FileNotFoundError:
    raise FileNotFoundError(f'The specified .csv is not at {csv_path}')
else:
    for _, dicom_file in enumerate(tqdm(dicom_files)):
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, struct.error):
            shutil.move(dicom_file,
                        os.path.join(options.base_dir, 'rejected_files'))
        else:
            _specific_sort(dicom_file=dicom_file,
                           dest_dir=options.project_dir,
                           cohort_list=cohort_list,
                           ds=ds) 
