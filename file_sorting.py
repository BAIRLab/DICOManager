#!/usr/bin/python
# %%
import pandas as pd
import pydicom
import os
import shutil
import glob
import struct
import csv
import optparse
from tqdm import tqdm


# %%
__author__ = ["Evan Porter", "Ron Levitin", "Nick Myziuk"]
__liscense__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


# Alter this path for directories not stored with '/data'
parser = optparse.OptionParser()

# parser.add_option('-a', '--alt', action='store',
                #   dest='alt', help='Alternative source dir', default=None)
parser.add_option('-m', '--move', action='store_true', dest='move_file',
                  help='Move instead of default of copy', default=False)
parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory containing unsorted files', default='/data/imported_data/')
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN csv to sort from, should be located in -b', default=None)
parser.add_option('-p', '--project-dest', action='store', dest='project_dir',
                    default = '/data/sorted_data/')
parser.add_option('-d', '--date', action='store_true', dest='date',
                  help='sort modalities by date', default=False)

options, args = parser.parse_args()

if not options.csv_file:
    raise NameError('A sorting csv file must be specified with flag --csv')

dicom_files = glob.glob(os.path.join(options.base_dir, '**/*.dcm*'), recursive=True)

print(f"base_dir: {options.base_dir}\n# of dicoms: {len(dicom_files)}")

def _write_to_path(dicom_file, dest_dir, patientID, subfolder=False):
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
    path_list = [dest_dir, patientID]
    
    print(dicom_file)

    if subfolder:
        path_list.append(subfolder)

    new_path = os.path.join(*path_list)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    _, filename = os.path.split(dicom_file)
    destination = os.path.join(new_path, filename)
    
    if options.move_file:
        count+=1
        shutil.move(dicom_file, destination)
    else:
        shutil.copy(dicom_file, destination)


def _specific_sort(dicom_file, dest_dir, cohort_list,
                #    patientID, modality, 
                ds):
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
    # with open(os.path.join('modality.csv'), mode='r') as csv_file:
    #     input_file = csv.reader(csv_file)
    #     subfolders = dict((rows[0], rows[1]) for rows in input_file)

    write_params = {"dest_dir": dest_dir,
                    "patientID": ds.PatientID,
                    "dicom_file": dicom_file,
                    # "date": None
                    }

    # if hasattr(ds, 'AcquisitionDate'):
        # write_params['date'] = ds.AcquisitionDate
    # elif hasattr(ds, 'StudyDate'):
        # write_params['date'] = ds.StudyDate
    # print(f"{ds.PatientID}: {cohort_list}\n")
    try:
        if int(ds.PatientID) in cohort_list or ds.PatientID in str(cohort_list):
            # print("Found a patient")
            if "CBCT" in ds.StudyDescription:
                subfolder = "CBCT"
            else:
                subfolder = ds.Modality
            write_params.update({"subfolder": subfolder})
            _write_to_path(**write_params)
    except ValueError:
        pass

# Command line starts below here #
count = 0
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

print(count)
            # # if "StudyDescription" not in dir(ds):
            #     # ds.add_new([0x0008, 0x1030], 'LO', '')
            # if "Modality" in dir(ds):
            #     # project_dir = options.project_dir
            #     if ds.StudyDescription[:4] == 'CBCT':
            #         _specific_sort(dicom_file=dicom_file,
            #                        file_path=project_dir,
            #                        cohort_list=cohort_list,
            #                     #    patientID=ds.PatientID,
            #                        modality='CBCT',
            #                        ds=ds)
            #     else:
            #         _specific_sort(dicom_file=dicom_file,
            #                        file_path=project_dir,
            #                        cohort_list=cohort_list,
            #                     #    patientID=ds.PatientID,
            #                        modality=ds.Modality,
            #                        ds=ds)
