import os
import shutil
import pydicom
from anytree import NodeMixin
from anytree.iterators.levelorderiter import LevelOrderIter
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import warnings


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorwarn(message: str):
    """[Fancy warning in color]

    Args:
        message (str): [Warning message to display]

    References:
        https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    """
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        relative = '/'.join(Path(filename).parts[-2:])
        return f'{relative}: line {lineno}: {category.__name__}: {message} \n'

    warnings.formatwarning = warning_on_one_line
    warnings.warn(bcolors.WARNING + message + bcolors.ENDC)


def save_tree(tree: NodeMixin, path: str, prefix: str = 'group') -> None:
    """[saves copy of dicom files to a specified location, ordered
        the same as the tree layout]

    Args:
        tree (NodeMixin): [tree to save]
        path (str): [absolute path to write the file tree]
        prefix (str, optional): [Specifies directory prefix as
            'group', 'date' or None]. Default to 'group'.

    Notes:
        prefix = 'date' not functional
    """
    if path[:-1] == '/':
        path = path[:-1]

    try:
        prefix = prefix.lower()
    except Exception:
        pass

    treeiter = LevelOrderIter(tree)
    for index, node in enumerate(treeiter):
        if prefix == 'group':
            names = [p.dirname for p in node.path]
        elif prefix == 'date':
            names = [p.datename for p in node.path]
        else:
            names = [p.name for p in node.path]

        subdir = path + '/'.join(names) + '/'
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        if repr(node) == 'Modality':
            for key in node.data:
                for fname in node.data[key]:
                    original = fname.filepath
                    newpath = subdir + fname.name
                    shutil.copy(original, newpath)
    print(f'\nTree {tree.name} written to {path}')


def current_datetime() -> str:
    dateobj = datetime.now()
    dateobj = dateobj.replace(microsecond=0)
    str_name = dateobj.isoformat().replace(':', '.')
    return str_name


def mod_getter(modtype: str) -> object:
    """[decorator to yield a property getter for the
        specified modality type]

    Args:
        modtype (str): [the specified modality type]

    Returns:
        object: [a property getter function]
    """
    def func(self):
        modlist = []
        if modtype in self.data:
            modlist.append(self.data[modtype])
        return sorted(modlist)
    return func


# Move to tools
def print_rts(rts):
    """[Simplified printing of DICOM RTSTRUCT without referenced UIDs]

    Args:
        rts ([pathlib.Path or pydicom.dataset]): [DICOM RTSTRUCT path or dataset]
    """
    if type(rts) is not pydicom.dataset:
        rts = pydicom.dcmread(rts)

    del rts[(0x3006, 0x0080)]
    del rts[(0x3006, 0x0010)]
    del rts[(0x3006, 0x0020)]
    del rts[(0x3006, 0x0039)]
    print(rts)


@dataclass
class VolumeDimensions:
    dicoms: list
    origin: list = None
    rows: int = None
    cols: int = None
    slices: int = None
    dims: list = None
    dx: float = None
    dy: float = None
    dz: float = None
    ipp: list = None
    vox_size: list = None
    flipped: bool = False
    multi_thick: bool = False

    def __post_init__(self):
        if 'RTDOSE' in self.dicoms:
            filepath = self.dicoms['RTDOSE'][0].filepath
            ds = pydicom.dcmread(filepath)
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePositionPatient
            self.dz = float(ds.SliceThickness)
        else:
            if 'CT' in self.dicoms:
                files = self.dicoms['CT']
            elif 'MR' in self.dicoms:
                files = self.dicoms['MR']

            ds = self._calc_n_slices(files)

        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = map(float, ds.PixelSpacing)
        self.dims = [self.rows, self.cols, self.slices]
        self.position = ds.PatientPosition
        self.vox_size = [self.dx, self.dy, self.dz]
        self.dicoms = None

    def _calc_n_slices(self, files: list):
        """[calculates the number of volume slices]

        Args:
            files ([DicomFile]): [A list of DicomFile objects]

        Notes:
            Creating the volume by the difference in slice location at high and
            low instances ensures proper registration to rstructs, even if
            images slices are missing. We can interpolate to the lowest
            instance if we do not have instance 1, but we cannot extrapolate
            if higher instances are missing
        """
        inst0 = np.inf
        inst1 = -np.inf
        slice_thicknesses = []

        for dcm in files:
            ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
            self.ipp = ds.ImagePositionPatient
            self.iop = ds.ImageOrientationPatient
            inst = int(ds.InstanceNumber)
            slice_thicknesses.append(float(ds.SliceThickness))
            if inst < inst0:  # Low instance
                inst0 = inst
                z0 = float(self.ipp[-1])
            if inst > inst1:  # High instance
                inst1 = inst
                z1 = float(self.ipp[-1])
            self.zlohi = (z0, z1)

        if inst0 > 1:
            z0 -= ds.SliceThickness * (inst0 - 1)
            inst0 = 1

        slice_thicknesses = list(set(slice_thicknesses))
        if len(slice_thicknesses) > 1:
            self.multi_thick = True

        self.dz = min(slice_thicknesses)
        self.origin = np.array([*self.ipp[:2], min(z0, z1)])
        self.slices = 1 + round((max(z0, z1) - min(z0, z1)) / self.dz)

        if z1 > z0:
            # TODO: We can replace thiw with the ImagePositionPatient header
            self.flipped = True

        return ds

    @property
    def shape(self):
        return (self.rows, self.cols, self.slices)

    def coordrange(self):
        pts_x = self.origin[0] + np.arange(self.rows) * self.dx
        pts_y = self.origin[1] + np.arange(self.cols) * self.dy
        pts_z = self.origin[2] + np.arange(self.slices) * self.dz
        if self.flipped:
            pts_z = pts_z[..., ::-1]
        return (pts_x, pts_y, pts_z)

    def coordgrid(self):
        pts_x, pts_y, pts_z = self.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)

    def Mgrid(self, dcm_hdr):
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
        # Unpacking arrays is poor form, but I'm feeling rebellious...
        X_x, X_y, X_z = np.array(IOP[:3]).T
        Y_x, Y_y, Y_z = np.array(IOP[3:]).T
        S_x, S_y, S_z = np.array(IPP)
        D_i, D_j = self.dx, self.dy
        i, j = np.indices((self.rows, self.cols))

        M = np.array([[X_x*D_i, Y_x*D_j, 0, S_x],
                      [X_y*D_i, Y_y*D_j, 0, S_y],
                      [X_z*D_i, Y_z*D_j, 0, S_z],
                      [0, 0, 0, 1]])

        C = np.array([i, j, 0, 1])

        return np.rollaxis(np.stack(np.matmul(M, C)), 0, 3)


def check_dims(func):
    def wrapped(cls, modality, *args, **kwargs):
        if not hasattr(cls, 'dims'):
            if modality.name in ['CT', 'MR']:
                cls.dims = VolumeDimensions(modality.data)
        return func(cls, modality, *args, **kwargs)
    return wrapped
