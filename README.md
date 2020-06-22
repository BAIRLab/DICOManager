# DICOM Pre-processing Library

## Purpose
To take DICOM files, imported to /data/imported_data (or another specified location)
and sorts based by provided MRN list.

User can then reconstruct volumes with the reconstruct.py functions.

Future support for inline, post-stored automatic reconstruction will be added.

## Prequisites
### Packages
Package requirements are specified in requirements.txt
These requirements can be installed with
```
pip install -r requirements.txt
```

### File Tree Structure
This libary is designed to function with the following directory tree. Any
alterations will require changing directory locations within file_sorting.py.

Any non-DICOM files, or those with corrupted headers will be moved to the
'rejected_files' director

Future versions may include a script to create the proper file tree in
linux / unix systems.

```
data / base directory
├── imported_data
│   └──  <file>.dcm
├── DICOManager
|   ├── <project>.csv
│   ├── modality.csv
│   ├── reconstruction.py
|   └── file_sorting.py
├── rejected_files
|   └── *.dcm
└── sorted_data
    └── <project>
        └── MRN0
            ├── MODAILITY0
            └── MODAILITY1
                └── *.dcm
 ```

## Project Overview
### requirements.txt
Required libraries for pip install, see above for guide

### file_sorting.py
Initally imported data of just DICOM files will be dumped into imported_data.
From which, sorting cam be completed for a given <project>. Sorting is completed
via a <project>.csv of MRN values and sorts them by MRN and date (if desired),
and then modality. It is recommended for PHI confidentiality, that MRNs are
replaced by anonymously coded numbers per patient.

Modalities are chosen from the modality.csv. Unique encodings can be provided,
with mapping of first row to directories of the second row.

Modalities will be choosen from the standard DICOM molalities, unless 'CBCT' is
in the SeriesDesecription, in which the .dcm files will be stored under CBCT.

Parsed arguments for this function include:
```
-b, --base: str (Default : pwd)
    Specify the base directory that sorting is occuring.  
-c, --csv: str
    Specify the path to a .csv file contained within sort_csv directory
-m, --move: bool
    Specify if the dicom files are moved or copied to the sorted directory
-p, --project-dest: str (Default : '/data/sorted_data/')
    Specify the location of the sorted dicom directory
```


### recon_sorted.py
This function is a script to apply the reconstruction.py functions to a
sorted project directory.

Parsed arguements for this function include:
```
-b, --base: str
        A path to the sorted project directory
-c, --csv: str
        A path to a .csv file in the format of example.csv, indicating
        the MRN values to be reconstructed
-d, --dest_dir: str
        A path to a file where the final .npy volumes will be stored
-j, --json: str
        A path to a .json file for the contour name dictionary to
        map contour names to a common name
-p, --project_name: str
        A string representing the name to append to the front of the
        saved .npy volume
```

### clean_rtstructs.py
If specified, this function will move all but the newest RTSTRUCT from a
sorted patient directory for simpler management of redunant outdate rt files.
If specified the remaining strutures and be printed.

Parsed arguements for this function include:
```
-b, --base: str
        A path to the sorted project directory
-c, --csv: str
        A path to a .csv file in the format of example.csv, indicating
        the MRN values to be reconstructed
-d, --dest_dir: str
        A path to a file where the final .npy volumes will be stored
-j, --json: str
        A path toa .json file for the contour name dictionary to
        map contour names to a common name
-s, --summary: bool
        Prints the names of the remaining RTSTRUCT ROIs
-v, --verbose: bool
        Prints the files and their relocated path
-r, --read_only: bool
        Only lists the ROIs in the RTSTRUCTs in the base directory
```

### reconstruction.py
Reconstructs PET, CT, CBCT, RTSTRUCT, RTDOSE DICOM formats into float32 numpy
arrays of original coordinate systems. Each function takes a specified list of
patient .dcm files of a given modality and returns a reconstructed volume.

#### PET: pet
Calculates the time corrected SUVbw PET value for the registered CT coordinate
system. Returns a numpy array of float32 values.

#### CBCT / CT : ct
Both CBCT and CT perform similarly, they are simply stored under different names.
Reconstuction is done at original image coordinates. Future work will include
projection of CBCT into CT coordinate space.

#### DOSE : dose
Reconstruction of Pinnacle 11.0 dose files into registered CT coordinate space.

#### RTSTRUCT : struct
RTSTRUCT files are saved as a list of arrays, but the dimensions are
(number-of-masks, x, y, z). Each element in the arrays are boolean. Masks are
returned in order as specified, except in cases where mask is not present in
the RTDOSE file.

#### MRI: mri
Creates an MRI volume and returns a numpy array of float32 values.
