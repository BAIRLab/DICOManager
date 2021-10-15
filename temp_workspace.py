import anytree
import csv
import functools
import multiprocessing
import numpy as np
import os
import pydicom
import shutil
import time
import warnings
from anytree import NodeMixin
from scipy import ndimage
from anytree.iterators.levelorderiter import LevelOrderIter
from datetime import datetime
from pathlib import Path
from skimage import measure
from dataclasses import dataclass
from copy import copy
from concurrent.futures import ProcessPoolExecutor as ProcessPool


@dataclass
class VolumeDimensions:
    dicoms: list
    origin: list = None
    rows: int = None
    cols: int = None
    slices: int = None
    dx: float = None
    dy: float = None
    dz: float = None
    ipp: list = None
    flipped: bool = False
    multi_thick: bool = False

    def __post_init__(self):
        self._compute_dims()

    def _dims_from_grid(self, grid):
        grid_ndim = grid.ndim
        dicom_shape = np.zeros((grid_ndim-1))
        voxelsize = np.zeros((grid_ndim-1))
        for axis in range(grid_ndim-1):
            def fn(x):
                if x == axis:
                    return slice(0, None)
                return 0

            slices = tuple([fn(x) for x in range(grid_ndim-1)] + [axis])
            axis_ipp = grid[slices]
            vmin = np.min(axis_ipp)
            vmax = np.max(axis_ipp)

            if vmin == vmax:
                dicom_shape[axis] = len(axis_ipp)
                voxelsize[axis] = np.nan
            else:
                axis_ipp.sort()
                diffs = np.round(axis_ipp - np.roll(axis_ipp, 1), 3)[1:]
                nvox = 1 + (vmax - vmin) / np.min(diffs)
                dicom_shape[axis] = nvox
                voxelsize[axis] = np.min(diffs)
        return (np.round(dicom_shape), voxelsize)

    def _compute_dims(self):
        grids = []
        for dcm in self.dicoms:
            ds = pydicom.dcmread(dcm, stop_before_pixels=True)
            grids.append(self.Mgrid(ds, rectilinear=True)[..., :3])
        grids = np.rollaxis(np.array(grids), 0, 3)
        volume_shape, voxelsize = self._dims_from_grid(grids)
        print(voxelsize)
        voxelsize[np.isnan(voxelsize)] = ds.PixelSpacing
        print(voxelsize)

        self.rows = volume_shape[0]
        self.cols = volume_shape[1]
        self.slices = volume_shape[2]
        self.dx = voxelsize[0]
        self.dy = voxelsize[1]
        self.dz = voxelsize[2]
        self.position = ds.PatientPosition
        self.dicoms = None

    @property
    def voxel_size(self):
        return [self.dx, self.dy, self.dz]

    @property
    def shape(self):
        return [self.rows, self.cols, self.slices]

    @property
    def field_of_view(self):
        shape = np.array(self.shape)
        voxel_size = np.array(self.voxel_size)
        return shape * voxel_size

    def as_dict(self):
        temp = vars(self)
        if 'dicoms' in temp.keys():
            del temp['dicoms']
        return temp

    def coordrange(self):
        pts_x = self.origin[0] + np.arange(self.rows) * self.dx
        pts_y = self.origin[1] + np.arange(self.cols) * self.dy
        pts_z = self.origin[2] + np.arange(self.slices) * self.dz
        if self.flipped:
            pts_z = pts_z[..., ::-1]
        return [pts_x, pts_y, pts_z]

    def coordgrid(self):
        pts_x, pts_y, pts_z = self.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)

    def calc_z_index(self, loc):
        return int(round(abs((self.origin[-1] - loc) / self.dz)))

    def Mgrid(self, dcm_hdr, rectilinear=True):
        """
        Function
        ----------
        Given a DICOM CT image slice, returns an array of pixel coordinates

        Returns
        ----------
        numpy.ndarray
            A numpy array of shape Mx2 where M is the dcm.Rows x dcm.Cols,
            the number of (x, y) pairs representing coordinates of each pixel

        Notes
        ----------
        Computes M via DICOM Standard Equation C.7.6.2.1-1
            https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037
        """
        IOP = dcm_hdr.ImageOrientationPatient
        IPP = dcm_hdr.ImagePositionPatient
        if rectilinear:
            IOP = np.round(IOP)
        # Unpacking arrays is poor form, but I'm feeling rebellious...
        X_x, X_y, X_z = np.array(IOP[:3]).T
        Y_x, Y_y, Y_z = np.array(IOP[3:]).T
        S_x, S_y, S_z = np.array(IPP)
        D_i, D_j = dcm_hdr.PixelSpacing
        i, j = np.indices((dcm_hdr.Rows, dcm_hdr.Columns))

        M = np.array([[X_x*D_i, Y_x*D_j, 0, S_x],
                      [X_y*D_i, Y_y*D_j, 0, S_y],
                      [X_z*D_i, Y_z*D_j, 0, S_z],
                      [0, 0, 0, 1]])
        C = np.array([i, j, np.zeros_like(i), np.ones_like(j)]).T
        return np.rollaxis(np.tensordot(M, C, axes=([1], [2])), 0, 3)

    def resampled_update(self, ratio: float) -> None:
        # Updates the volume dimensions when the volume is resampled
        if type(ratio) is int or type(ratio) is float:
            ratio = np.array([ratio, ratio, ratio])
        if type(ratio) is list:
            ratio = np.array(ratio)

        self.dx /= ratio[0]
        self.dy /= ratio[1]
        self.dz /= ratio[2]

        self.rows = int(round(self.rows * ratio[0]))
        self.cols = int(round(self.cols * ratio[1]))
        self.slices = int(round(self.slices * ratio[2]))

    def crop_update(self, values: list) -> None:
        # Updates the volume dimensions when volume is cropped
        xlo, xhi, ylo, yhi, zlo, zhi = values.T.flatten()
        self.rows = xhi - xlo
        self.cols = yhi - ylo
        self.slices = zhi - zlo
