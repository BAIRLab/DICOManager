#!/usr/bin/python3

import os
from glob import glob as glob
import csv
import optparse
import json
from pathlib import Path
from dataclasses import dataclass
import reconstruction
import numpy as np

__author__ = "Evan Porter"
__license__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"

@dataclass
class FileName:
    """
    Function
    ----------
    Take a file path to a patient directory and parses out all modality
        specific DICOMs

    Parameters
    ----------
    fullpath : str
        A path to the given .dcm directory
    study_sorted : bool
        A boolean dictating if the directory is study sorted. To be depreciated

    Returns
    ----------
    None
    """
    fullpath: str
    project: str = None
    mrn: str = None
    date: str = None
    MR: list = None
    CT: list = None
    PET: list = None
    RTSTRUCT: list = None
    RTDOSE: list = None

    def __post_init__(self):
        # split_path = self.fullpath.split('/')
        self.project, _, self.mrn = self.fullpath.rpartition('/')

        # if len(split_path) == 3:
            # self.date = split_path[2]

        mod_list = ['MR', 'CT', 'PET', 'RTSTRUCT', 'RTDOSE']

        if self.date:
            path_list = [[self.project, self.mrn,
                          self.date, x, '*.dcm'] for x in mod_list]
        else:
            path_list = [[self.project, self.mrn, x, '*.dcm']
                         for x in mod_list]

        search_paths = [os.path.join(*x) for x in path_list]

        self.MR = glob(search_paths[0])
        self.CT = glob(search_paths[1])
        self.PET = glob(search_paths[2])
        self.RTSTRUCT = glob(search_paths[3])
        self.RTDOSE = glob(search_paths[4])

    def __repr__(self):
        mod_list = [self.MR, self.CT, self.PET, self.RTSTRUCT, self.RTDOSE]
        n_mr, n_ct, n_pet, n_rts, n_rtd = [len(x) for x in mod_list]
        return (f'{self.fullpath}:{n_mr}:{n_ct}:{n_pet}:{n_rts}:{n_rtd}')


def _file_namer(filename, modality):
    """
    Function
    ---------
    Names a file based on filepath

    Parameters
    ---------
    filename : filename object
        An initialized filename object
    modality : str
        A modality string or identifier

    Returns
    ----------
    current_name : str
        A name for the file to be saved
    """
    current_name = filename.project + '_'
    current_name += filename.mrn + '_'
    if filename.date:
        current_name += filename.date + '_'
    current_name += modality + '.npy'
    return current_name


usage = "usage: recon_sorted.py [opt1] ... \n Reconstructs from DICOM to Numpy arrays and saves in -d"
parser = optparse.OptionParser(usage)

parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory with sorted data to reconstruct', default=None)
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN list to reconstruct', default=None)
parser.add_option('-j', '--json', action='store', dest='contour_list',
                  help='Path to json of dictionary of RTSTRUCTS to reconstruct', default=None)
parser.add_option('-d', '--dest_dir', action='store', dest='dest_dir',
                  help='Directory to save numpy arrays', default=None)
 
options, args = parser.parse_args()

if not options.base_dir:
    raise NameError('A sorted project folder must be provided to reconstruct')

if not options.dest_dir:
    options.dest_dir = options.base_dir

# File tree aside from dicoms. Assumes nothing but dicoms in the sorted directories
file_tree = glob(os.path.join(options.base_dir, '**/*[!.dcm]'), recursive=True)

with open(options.csv_file, mode='r') as MRN_csv:
    filter_list = list(x[0] for x in csv.reader(MRN_csv))[1:]

print(f"filter_list: {filter_list}")
# Assumes df.split('/')[-2] is the patient MRN
pat_folders = list(set([df.rpartition('/')[0]
                        for df in file_tree if df.split('/')[-2] in filter_list]))

print(pat_folders)

for path in pat_folders:
    patient_group = FileName(path)
    print(patient_group)
    mr, ct, pet, rts, dose = [[] for _ in range(5)]

    if patient_group.MR:
        mr = reconstruction.mri(path)
        # print(f"MR Shape: {mr.shape}")
        mr_name = _file_namer(patient_group, 'MR')
        # if not options.pool:
            # np.save(mr_name, mr)
    if patient_group.CT:
        ct = reconstruction.ct(path)
        ct_name = _file_namer(patient_group, 'CT')
        # if not options.pool:
            # np.save(ct_name, ct)
    if patient_group.PET:
        pet = reconstruction.pet(path)
        pet_name = _file_namer(patient_group, 'PET')
        # if not options.pool:
            # np.save(pet_name, pet)
    if patient_group.RTSTRUCT:
        if not options.contour_list:
            raise NameError('A .csv of contours must be specified with -l')
        with open(options.contour_list, mode='r') as json_file:
            contour_list = json.load(json_file)
            print(f"contours: {contour_list}")
        rts = reconstruction.struct(path, contour_list)
        rts_name = _file_namer(patient_group, 'RTSTRUCT')
        # if not options.pool:
            # np.save(rts_name, rts)
    if patient_group.RTDOSE:
        dose = reconstruction.dose(path)
        dose_name = _file_namer(patient_group, 'RTDOSE')
        # if not options.pool:
            # np.save(dose_name, dose)
    # if options.pool:
    pool_dict = {'MR': mr,
                    'CT': ct,
                    'PET': pet,
                    'RTSTRUCT': rts,
                    'RTDOSE': dose
                    }
    print(f"MR: {pool_dict['MR'].shape}\nRTS: {pool_dict['RTSTRUCT'].shape}")
    pool_name = _file_namer(patient_group, 'POOL')
    print(pool_name)
    np.save(pool_name, pool_dict)
