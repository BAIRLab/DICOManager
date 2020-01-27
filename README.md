# DICOM Processing Library

## Purpose
To take DICOM files, imported to /data/imported_data (or another specified location)
and sorts based by provided MRN list.

User can then reconstruct volumes with the reconstruct.py function.

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

```
parent directory
├── imported_data
|   ├── rejected_files
│   └── <file>.dcm
├── DICOManager
|   ├── <project>.csv
|   ├── <countour names>.json
|   ├── requirements.txt 
|   ├── clean_rtstructs.py 
│   ├── recon_sorted.py
│   ├── reconstruction.py
|   └── file_sorting.py
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
via a <project>.csv of MRN values and sorts them by MRN and then modality.
It is recommended for PHI confidentiality, that MRNs are replaced by anonymously
coded numbers per patient.

Any non-dicom files will be moved to ./rejected_files, to prevent read failures.

Modalities will be choosen from the standard DICOM molalities.

Parsed arguments for this function include:
```
-b, --base: str (Default: /data/imported_data/)
	A path to a directory containing the unsorted files to be sorted 
-c, --csv : str
	A path to a .csv file of the MRNs to be sorted. The .csv should be
	formatted like example.csv
-m, --move: bool (Default: False)
	A boolean to designate if the files should be move instead of
	copied to the project destination directory
-p, --project_dest: str (Default: /data/sorted_data)
	A path to a destination for the sorted files
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
patient .dcm files of a given modality and returns a reconstructed volume

#### PET: pet
Calculates the time corrected SUVbw PET value for the registered CT coordinate
system. Returns a numpy array of float32 values.

#### CBCT / CT : ct
Both CBCT and CT perform similarly, they are simply stored under different names.
Reconstuction is done at original image coordinates. Future work will include
projection of CBCT into CT coordinate space.

#### DOSE : dose
Reconstruction of Pinnacle 11.0 dose files into registered CT coordinate space

#### RTStruct : struct
RTStruct files are saved as a list of arrays, but the dimensions are
(number-of-masks, x, y, z). Each element in the arrays are boolean. Masks are
returned in order as specified, except in cases where mask is not present in
the RTDOSE file.
