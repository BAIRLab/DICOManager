# DICOM Pre-processing Library

Rewriting DICOManager to work without write-access. Designed to build a file tree and reconstruct patients entirely with a python package.

See tutorial.ipynb for a notebook tutorial on functionality and example.py for an example reconstruction function.

Reference branch v0.1 to see the prior version of DICOManager with file_sorting and reconstruction functions.

Remaining tasks:
* Support for data generators for pytorch / tensorflow
* Reading saved directory trees without re-sorting
* Updating deconstruction for new data structures
* Saving as either .npy or NIFTI format
* Improving documentation
* Formatting for pip install