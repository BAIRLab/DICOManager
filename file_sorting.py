#!/usr/bin/python

import glob
import optparse
import os
import shutil
import struct
import pydicom
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool

__author__ = ["Evan Porter", "Ron Levitin", "Nick Myziuk"]
__liscense__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


class sorting:
    """[class for parallelized mapping to sort DICOMs]
    """
    def __init__(self, cohort_list: list, project_dir: str):
        """[initialize sorting class instance]

        Args:
            cohort_list ([str OR int]): [list of mrn values]
            project_dir (str): [path to the working directory]
        """
        self.cohort_list = cohort_list
        self.options = options

    def sort_file(self, dicom_file: str):
        """[sorts and individual file]

        Args:
            dicom_file (str): [a path to a DICOM file]
        """
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, struct.error):
            shutil.move(dicom_file,
                        os.path.join(self.options.base_dir, 'rejected_files'))
        else:
            _specific_sort(dicom_file=dicom_file,
                           dest_dir=self.options.project_dir,
                           cohort_list=self.cohort_list,
                           ds=ds)


def _write_to_path(dicom_file: str, dest_dir: str,
                   patientID: [str, int], subfolder: bool = False):
    """[write a DICOM file to a path location]

    Args:
        dicom_file (str): [path to original DICOM location]
        dest_dir (str, int): [path to DICOM directory destination]
        patientID (str): [string of the mrn value]
        subfolder (bool, optional): [subfolder name to nest within
                                     mrn folder]. Defaults to False.
    """
    path_list = [dest_dir, patientID]

    if subfolder:
        path_list.append(subfolder)

        new_path = os.path.join(*path_list)

    if not os.path.exists(new_path):
        try:
            os.makedirs(new_path)
        except FileExistsError:
            pass

    _, filename = os.path.split(dicom_file)
    destination = os.path.join(new_path, filename)

    if options.move_file:
        shutil.move(dicom_file, destination)
    else:
        shutil.copy(dicom_file, destination)


def _specific_sort(dicom_file: str, dest_dir: str,
                   cohort_list: list, ds: pydicom.dataset.FileDataset):
    """[given a DICOM, sorts into the proper file structure]

    Args:
        dicom_file (str): []
        dest_dir (str): [description]
        cohort_list (list): [description]
        ds (pydicom.dataset.FileDataset): [description]
    """
    write_params = {"dest_dir": dest_dir,
                    "patientID": ds.PatientID,
                    "dicom_file": dicom_file,
                    }
    try:
        if match_id(ds.PatientID, cohort_list):
            if hasattr(ds, "StudyDescription"):
                if "CBCT" in ds.StudyDescription:
                    subfolder = "CBCT"
                else:
                    subfolder = ds.Modality
            else:
                subfolder = ds.Modality
            write_params.update({"subfolder": subfolder})
            _write_to_path(**write_params)
    except ValueError:
        pass


def match_id(ID: [str, int], cohort_list: list) -> bool:
    """[returns boolean if ID matches cohort_list]

    Args:
        ID (str OR int): [mrn string or int value]
        cohort_list ([str OR int]): [list of mrns in cohort]

    Returns:
        bool: [representing if ID is in cohort]
    """
    if ID in cohort_list:
        return True
    try:
        if ID in str(cohort_list):
            return True
    except Exception:
        if ID in int(cohort_list):
            return True
    except Exception:
        return False


# Command line starts below here
parser = optparse.OptionParser()

parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory containing unsorted files', default='/data/imported_data/')
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN csv to sort from, should be located in -b', default=None)
parser.add_option('-m', '--move', action='store_true', dest='move_file',
                  help='Move instead of default of copy', default=False)
parser.add_option('-p', '--project-dest', action='store', dest='project_dir',
                  default='/data/sorted_data/')

options, args = parser.parse_args()

if __name__ == '__main__':
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
        with ProcessPool() as P:
            fn = sorting(cohort_list, options)
            _ = list(tqdm(P.imap(fn.sort_file, dicom_files), total=len(dicom_files)))
