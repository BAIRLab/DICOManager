# DICOM Pre-processing Library

Rewriting DICOManager to work without write-access. Designed to build a file tree and reconstruct patients entirely with a python package.

Reference branch v0.1 to see the prior version of DICOManager with file_sorting and reconstruction functions.

Features to add:
* proper orientation based upon ImagePositionPatient matrix
* boolean mask to DICOM RTSTRUCT (moved forward from DICOManager v1)
* filtering (by mrn, studyUID, structure name, etc.)
* saving reconstructed arrays in directory tree
* image processing (window/level, crop, resample) with metadata and original information storage in header
* support for data generators for pytorch / tensorflow
* reading saved directory trees without re-sorting
* DICOM tag documentation