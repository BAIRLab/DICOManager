import os
import shutil
import pydicom
import anytree
from anytree import NodeMixin
from anytree.iterators.levelorderiter import LevelOrderIter
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import warnings
from copy import copy
import time
import csv
import functools
from concurrent.futures import ProcessPoolExecutor as ProcessPool
import multiprocessing


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


def colorwarn(message: str, source: str = None):
    """[Fancy warning in color]

    Args:
        message (str): [Warning message to display]

    References:
        https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    """
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        if source:
            relative = source
            return f'{relative}: {category.__name__}: {message} \n'
        else:
            relative = '/'.join(Path(filename).parts[-2:])
            return f'{relative}: line {lineno}: {category.__name__}: {message} \n'

    warnings.formatwarning = warning_on_one_line
    warnings.warn(bcolors.WARNING + message + bcolors.ENDC)


def save_tree(tree: NodeMixin, path: str, prefix: str = 'group',
              separate_volume_dir: bool = True) -> None:
    """[saves copy of dicom files (and volumes, if present) to a specified
        location, ordered the same as the tree layout]

    Args:
        tree (NodeMixin): [tree to save]
        path (str): [absolute path to write the file tree]
        prefix (str, optional): [Specifies directory prefix as
            'group', 'date' or None]. Default to 'group'.
        separate_volume_dir (bool, optional): [Seperates volumes and dicoms
            at the modality level]. Default to 'True'.

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

    has_volumes = tree.has_volumes()
    has_dicoms = tree.has_dicoms()

    for index, node in enumerate(treeiter):
        if prefix == 'group':
            names = [p.dirname for p in node.path]
        elif prefix == 'date':
            names = [p.datename for p in node.path]
        else:
            names = [p.name for p in node.path]

        if separate_volume_dir and repr(node) == 'Cohort':
            volpath = path + '/'.join(names) + '_volumes/'
            dcmpath = path + '/'.join(names) + '_dicoms/'
        elif separate_volume_dir:
            vnames = copy(names)
            dnames = copy(names)
            vnames[0] += '_volumes'
            dnames[0] += '_dicoms'
            volpath = path + '/'.join(vnames) + '/'
            dcmpath = path + '/'.join(dnames) + '/'
        else:
            volpath = path + '/'.join(names) + '/'
            dcmpath = path + '/'.join(names) + '/'

        if not os.path.isdir(volpath) and has_volumes:
            os.mkdir(volpath)
        if not os.path.isdir(dcmpath) and has_dicoms:
            os.mkdir(dcmpath)

        if repr(node) == 'Modality':
            for key in node.volumes_data:
                for fname in node.volumes_data[key]:
                    for volume in node.volumes_data[key]:
                        newpath = volpath + fname.SeriesUID
                        np.save(newpath, volume.export())
            for key in node.dicoms_data:
                for fname in node.dicoms_data[key]:
                    original = fname.filepath
                    newpath = dcmpath + fname.name
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
        if modtype in self.dicoms_data:
            modlist.append(self.dicoms_data[modtype])
        return sorted(modlist)
    return func


def mod_setter(modtype: str) -> object:
    def func(self, item: object):
        if modtype not in self.dicoms_data:
            self.dicoms_data.update({modtype: {}})
        else:
            self.dicoms_data[modtype].update({item.SeriesUID: item})
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

        z0, z1 = (0, 0)
        #for ds in files:
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
        self.origin = np.array([*self.ipp[:2], max(z0, z1)])
        self.slices = 1 + round((max(z0, z1) - min(z0, z1)) / self.dz)

        if z1 < z0:
            # TODO: We can replace thiw with the ImagePositionPatient header
            self.flipped = True

        return ds

    @property
    def shape(self):
        return (self.rows, self.cols, self.slices)

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
        C = np.array([i, j, np.zeros_like(i), np.ones_like(i)])

        return np.rollaxis(np.stack(np.matmul(M, C)), 0, 3)


def check_dims(func):
    def wrapped(cls, modality, *args, **kwargs):
        if not hasattr(cls, 'dims'):
            if modality.name in ['CT', 'MR']:
                cls.dims = VolumeDimensions(modality.data)
        return func(cls, modality, *args, **kwargs)
    return wrapped


def three_axis_plot(array: np.ndarray, name: str, mask: np.ndarray = None) -> None:
    import matplotlib.pyplot as plt

    shape = array.shape
    coronal = array[shape[0]//2, :, :]
    sagittal = array[:, shape[1]//2, :]
    axial = array[:, :, shape[2]//2]

    ax0 = plt.subplot(3, 1, 1)
    ax0.imshow(sagittal.T, cmap='binary_r')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(3, 1, 2)
    ax1.imshow(coronal.T, cmap='binary_r')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(3, 1, 3)
    ax2.imshow(axial, cmap='binary_r')
    ax2.set_xticks([])
    ax2.set_yticks([])

    if mask is not None:
        ma_arr = np.ma.masked_where(mask == 0, mask)
        ma_cor = ma_arr[shape[0]//2, :, :]
        ma_sag = ma_arr[:, shape[1]//2, :]
        ma_ax = ma_arr[:, :, shape[2]//2]
        print('masked arrays created')

        ax0.imshow(ma_sag.T, cmap='bwr', alpha=0.8)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1.imshow(ma_cor.T, cmap='bwr', alpha=0.8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.imshow(ma_ax, cmap='bwr', alpha=0.8)
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.savefig(name+'.png', format='png', dpi=300, bbox_inches='tight')


def dict_to_dataclass(d: dict, name: str = 'd_dataclass') -> object:
    """[Converts a dictionary into a dataclass]

    Args:
        d (dict): [Dict of name and properties for the dataclass]
        name (str, optional): [Name of dataclass]. Defaults to 'd_dataclass'.

    Returns:
        [object]: [Created dataclass]
    """
    @dataclass
    class Wrapped:
        __annotations__ = {k: type(v) for k, v in d.items()}

    Wrapped.__qualname__ = Wrapped.__name__ = name

    dclass = Wrapped(**d)
    return dclass


def clear_runtime():
    f = open('runtimes.csv', "w+")
    f.close()


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        with open('runtimes.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([elapsed_time])
        return value
    return wrapper_timer


def average_runtime():
    with open('runtimes.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        data = np.array(list(reader), dtype=np.float)
        print(f'Average runtime: {np.mean(data):0.3f} +/- {np.std(data):0.3f} seconds')


def split_tree(primary: NodeMixin, n: int = 10) -> list:
    """[Splits a tree into a series of n-sized trees]

    Args:
        primary (NodeMixin): [Primary tree to split]
        n (int, optional): [Size of each smaller tree]. Defaults to 10.

    Returns:
        list: [A list of the split trees]
    """
    trees = []
    tree = primary.__class__(primary.name)
    count = 0
    while len(primary):
        child = primary.pop()
        tree.adopt(child)
        count += 1
        if count == n:
            trees.append(tree)
            tree = primary.__class__(primary.name)
            count = 0
    if len(tree) > 0:
        trees.append(tree)
    return trees


# this should really be type GroupUtils
def combine_trees(primary: NodeMixin, secondaries: list) -> NodeMixin:
    """[Combine a series of trees into the primary tree]

    Args:
        primary (NodeMixin): [Primary tree to return with secondaries]
        secondaries (list): [Secondaries to add to primary tree]

    Returns:
        NodeMixin: [Unified tree under type Primary]
    """
    for tree in secondaries:
        for child in tree:
            primary.adopt(child)
    return primary


def insert_into_tree(tree: NodeMixin, mod_ptr_pairs: list) -> None:
    """[Inserts a volume pointer into a tree]

    Args:
        tree (NodeMixin): [Tree to insert volume pointers]
        mod_ptr_pairs (list): [list of modality and corresponding pointer]
    """
    for modality, pointer in mod_ptr_pairs:
        node = tree
        for a in modality.ancestors:
            node = anytree.search.find(node, filter_=lambda x: x.name == a.name)
        node._add_file(pointer)


def recon_fn(tree: NodeMixin) -> list:
    """[For reconstruction of a tree]

    Args:
        tree (NodeMixin): [Tree to reconstruct]

    Returns:
        [list]: [A list of tuples containing modality and ReconstructedFile]
    """
    return tree.recon(in_memory=False, return_mods=True)


def threaded_recon(primary: NodeMixin) -> NodeMixin:
    """[A multiprocessed reconstruction of primary]

    Args:
        primary (NodeMixin): [Tree for reconstruction]

    Returns:
        NodeMixin: [Primary with the volume pointers inserted]

    Notes:
        Reconstruction does not scale with processors, so we split the
            tree into smaller trees, reconstruct each in parallel and
            then recombine the trees
    """
    trees = split_tree(primary, n=10)

    ncpus = multiprocessing.cpu_count()

    with ProcessPool(max_workers=ncpus//4) as P:
        results = list(P.map(recon_fn, trees))

    primary = combine_trees(primary, trees)

    for mod_ptr_pairs in results:
        insert_into_tree(primary, mod_ptr_pairs)

    return primary
