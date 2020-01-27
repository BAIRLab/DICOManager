#!/usr/bin/python3

import os
from glob import glob as glob
import csv
import optparse
import json
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
import reconstruction
import numpy as np

__author__ = ["Evan Porter", "Ron Levitin"]
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
    MR: list = None
    CT: list = None
    PET: list = None
    RTSTRUCT: list = None
    RTDOSE: list = None

    def __post_init__(self):
        self.project, _, self.mrn = self.fullpath.rpartition('/')
        
        mod_list = ['MR', 'CT', 'PET', 'RTSTRUCT', 'RTDOSE']

        path_list = [[self.project, self.mrn, x, '*.dcm']
                        for x in mod_list]
        
        if options.project:
            self.project = options.project

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


usage = "usage: recon_sorted.py [opt1] ... \n Reconstructs from DICOM to Numpy arrays and saves in -d"
parser = optparse.OptionParser(usage)

parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory with sorted data to reconstruct', default=None)
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN list to reconstruct', default=None)
parser.add_option('-d', '--dest_dir', action='store', dest='dest_dir',
                  help='Directory to save numpy arrays', default=None)
parser.add_option('-j', '--json', action='store', dest='contour_list',
                  help='Path to json of dictionary of RTSTRUCTS to reconstruct', default=None)
parser.add_option('-p', '--project_name', action='store', dest='project',
                    help='Project name to prepend to files', default=None)
 
options, args = parser.parse_args()

if not options.base_dir:
    raise NameError('A sorted project folder must be provided to reconstruct')

if not options.dest_dir:
    options.dest_dir = options.base_dir

file_tree = glob(os.path.join(options.base_dir, '**/*[!.dcm]'), recursive=True)

with open(options.csv_file, mode='r') as MRN_csv:
    filter_list = list(x[0] for x in csv.reader(MRN_csv))[1:]

pat_folders = list(set([df.rpartition('/')[0]
                        for df in file_tree if df.split('/')[-2] in filter_list]))

for path in tqdm(pat_folders):
    patient_group = FileName(path)

    mr, ct, pet, rts, dose = [[] for _ in range(5)]

    if patient_group.MR:
        mr = reconstruction.mri(path)
    if patient_group.CT:
        ct = reconstruction.ct(path)
    if patient_group.PET:
        pet = reconstruction.pet(path)
    if patient_group.RTSTRUCT:
        if not options.contour_list:
            raise NameError('A .csv of contours must be specified with -l')
        with open(options.contour_list, mode='r') as json_file:
            contour_list = json.load(json_file)
        rts = reconstruction.struct(path, contour_list)
    if patient_group.RTDOSE:
        dose = reconstruction.dose(path)
    pool_dict = {'MR': mr,
                 'CT': ct,
                 'PET': pet,
                 'RTSTRUCT': rts,
                 'RTDOSE': dose
                }
    if not os.path.exists(options.dest_dir):
        os.makedirs(options.dest_dir)

    np.save(Path(options.dest_dir) / 
            (patient_group.project + '_' + patient_group.mrn),
            pool_dict)
