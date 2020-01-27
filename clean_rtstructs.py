#!/usr/bin/python

import optparse
import os
import pydicom
import json
from glob import glob
import shutil
from datetime import datetime
import time
import pandas as pd
import warnings
from pathlib import Path


def _get_structs(struct_path, contours=None):
    """
    _get_structs Given a path to an RTSTRUCT DICOM, return a list of contour names.
    
    Args:
        struct_path (str): path to DICOM
    """

    ds = pydicom.dcmread(struct_list[0])
    struct_names = []
    
    for contour in ds.StructureSetROISequence:
        name = contour.ROIName.lower()
        if contours:
            for key in contours:
                clist = [x.lower() for x in contours[key]]
                if name == key.lower() or name in clist:
                    struct_names.append(key.lower())
        else:
            struct_names.append(name)
    
    struct_names.sort()

    return struct_names

def _creation_posix(filepath):
    """
    Function
    ----------
    Returns the POSIX time for the Instance Creation of DICOM 'f' 

    Parameters
    ----------
    filepath : str
        A path location of the DICOM file

    Returns
    ----------
    time : float
        The POSIX time for the Instance Creation of the DICOM
    """
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    ds_date = ds.InstanceCreationDate
    ds_time = ds.InstanceCreationTime
    dt = datetime.strptime(ds_date + ds_time, "%Y%m%d%H%M%S.%f")
    return time.mktime(dt.timetuple())


class _bcolors:
    """
    Reference
    ----------
    https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


usage = "usage: clean_rtstructs.py [opt1] ... \n Moves old RTSTRUCTs to -d"
parser = optparse.OptionParser(usage)

parser.add_option('-b', '--base', action='store', dest='base_dir',
                  help='Directory with sorted data to clean', default=None)
parser.add_option('-c', '--csv', action='store', dest='csv_file',
                  help='MRN list of patient files to clean', default=None)
parser.add_option('-d', '--dest_dir', action='store', dest='dest_dir',
                  help='Directory to move extra RTSTRUCTs', default=None)
parser.add_option('-j', '--json', action='store', dest='contour_list',
                  help='Path to json of dictionary of RTSTRUCTS for summary', default=None)
parser.add_option('-s', '--summary', action='store_true', dest='summary',
                  help='Prints remaining ROI Names for RTSTRUCTs in -b', default=False) 
parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                  help='Prints moved files', default=False)
parser.add_option('-r', '--read_only', action='store_true', dest='read_only',
                  help='Use flag to interrogate RTSTRUCTS without moving', default=False)
options, args = parser.parse_args()

if not options.base_dir:
    raise NameError('A sorted DICOM directory must be specified by --base')
if not options.read_only and not options.dest_dir:
    raise NameError('A destination directory must be specified by --dest_dir')

if options.csv_file:
    mrns = pd.read_csv(options.csv_file, engine='python')['MRN'].tolist()
else:
    mrns = os.listdir(options.base_dir)

mrns.sort()

if not options.read_only:
    for mrn in mrns:
        struct_list = glob(os.path.join(options.base_dir, mrn, 'RTSTRUCT/*.dcm'))
        if len(struct_list) > 1:
            times = [_creation_posix(struct) for struct in struct_list] 
            _ = struct_list.pop(times.index(max(times)))
            
            for struct in struct_list:
                dest = os.path.join(options.dest_dir, mrn)
                if not os.path.exists(dest):
                    os.makedirs(dest)
                if options.verbose:
                    print(f'{struct} -> {dest}') 
                shutil.move(struct, dest)

if options.summary or options.contour_list:
    try:
        with open(options.contour_list, 'r') as json_file:
            contours = json.load(json_file)
        print('\n\n')
    except:
        contours = []
        print(f'\n{_bcolors.WARNING}No .json, will print raw ROI names{_bcolors.ENDC}\n')
    print(f'        MRN   : # :   Contour Names')
    
    for mrn in mrns:
        struct_list = glob(os.path.join(options.base_dir, mrn, 'RTSTRUCT/*.dcm'))
        if options.read_only:
            count_files = []
            for i, f in enumerate(struct_list):
                count_files.append(_get_structs(f, contours))
                if i != 0:
                    print(f'              : {i} : {count_files[i]}')
                else:
                    print(f'{mrn:>13s} : {i} : {count_files[i]}')
        else:
            print(f'{mrn} : {_get_structs(struct_list[0], contours)}')