# Data Pre-processing Library

## Purpose
To take DICOM files, imported to /data/imported_data and sorts based by provided
MRN list. Then volumes can be reconstructed with the reconstruct.py functions.

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
|   ├── sort_csv
|   |   └── <project>.csv
│   ├── modality.csv
│   ├── reconstruction.py
|   └── file_sorting.py
├── rejected_files
|   └── *.dcm
└── sorted_data
    └── <project>
        └── MRN0
            └── DATE0 (Optional)
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

Parsed arguments for this function include:
-d, --date: bool
    Specify if sorting below MRN should include date before modality.
    Hierarchy is AcquisitionDate then StudyDate
-b, --base: str (Default : pwd)
    Specify the base directory that sorting is occuring.  
-f, --file: str
    Specify the name of a .csv file contained within sort_csv directory

### reconstruction.py
Reconstructs PET, CT, CBCT, RTSTRUCT, RTDOSE DICOM formats into float32 numpy
arrays of original coordinate systems. Each function takes a specified list of
patient .dcm files of a given modality and returns a reconstructed volume

#### CBCT / CT : ct_reconstruction
Both CBCT and CT perform similarly, they are simply stored under different names.
Reconstuction is done at original image coordinates. Future work will include
projection of CBCT into CT coordinate space.

#### DOSE : dose_reconstruction
Reconstruction of Pinnacle 11.0 dose files into registered CT coordinate space

#### RTStruct : struct_reconstruction
RTStruct files are saved as a list of arrays, but the dimensions are
(number-of-masks, x, y, z). Each element in the arrays are boolean. Masks are
returned in order as specified, except in cases where mask is not present in
the RTDOSE file.
