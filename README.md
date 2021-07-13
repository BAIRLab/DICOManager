# DICOM Pre-processing Library

## Installation
For the required packages to use this toolkit, please refer to requirements.txt. These can be batch installed with pip using `pip install -r requirements.txt`. Python version >= 3.8 is recommended for the local environmant.

## Overview
A DICOM management toolkit designed around the construction and operation upon DICOM file trees. File groupings can be created using any of the grouping classes. The heirarchy used in this library is:

1. Cohort
2. PatientID
3. FrameOfReferenceUID
4. StudyInstanceUID
5. SeriesInstanceUID
6. Modality (contains individual dicom and reconstructed files)

## Resources
Reconstruction operations occur at the FrameOfReference level using the same coordiante system to reconstruct all files within the frame of reference. By default the library performs most operations using a multithreaded workflow with a bais towards storing data on disk. Function calls can be modified to store reconstructed volumes in memory if read-write access is limited on the system. For additional explination of functions and tools, please refer to tutorial.ipynb and an example workflow, please refer to example.py. The example workflow can be used via the call `python -m DICOManager.example`.

Note: This is currently a work in progress. Any isses encountered please submit an issue or pull request.

## Remaining Tasks
* Support for data generators for pytorch / tensorflow
* Reading saved directory trees without re-sorting
* Checking loaded tree for validitity
* Updating deconstruction for new data structures
* Saving as either .npy or NIFTI format or x-array
* Improving documentation and creation of wiki
* Formatting for pip install