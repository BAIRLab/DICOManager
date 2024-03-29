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
    """Fancy warning in color

    Args:
        message (str): Warning message to display

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
    """Saves copy of dicom files (and volumes, if present) to a specified
        location, ordered the same as the tree layout

    Args:
        tree (NodeMixin): Tree to save
        path (str): Absolute path to write the file tree
        prefix (str, optional): Specifies directory prefix as
            'group', 'date' or None. Default to 'group'.
        separate_volume_dir (bool, optional): Seperates volumes and dicoms
            at the modality level. Default to 'True'.

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
            os.makedirs(volpath, exist_ok=True)
        if not os.path.isdir(dcmpath) and has_dicoms:
            os.makedirs(dcmpath, exist_ok=True)

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
    print(f'Tree {tree.name} written to {path}')


def current_datetime() -> str:
    dateobj = datetime.now()
    dateobj = dateobj.replace(microsecond=0)
    str_name = dateobj.isoformat().replace(':', '.')
    return str_name


def mod_getter(modtype: str) -> object:
    """Decorator to yield a property getter for the
        specified modality type

    Args:
        modtype (str): The specified modality type

    Returns:
        object: A property getter function
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
    """Simplified printing of DICOM RTSTRUCT without referenced UIDs

    Args:
        rts (pathlib.Path or pydicom.dataset): DICOM RTSTRUCT path or dataset
    """
    if type(rts) is not pydicom.dataset:
        rts = pydicom.dcmread(rts)

    del rts[(0x3006, 0x0080)]
    del rts[(0x3006, 0x0010)]
    del rts[(0x3006, 0x0020)]
    del rts[(0x3006, 0x0039)]
    print(rts)


def dose_grid_coordrange(dosefile, ct_dims):
    ds = pydicom.dcmread(dosefile.filepath)
    dx, dy = ds.PixelSpacing
    cols = ds.Columns
    rows = ds.Rows
    origin = ds.ImagePositionPatient
    zlocs = ds.GridFrameOffsetVector
    pts_x = origin[0] + np.arange(rows) * dx
    pts_y = origin[1] + np.arange(cols) * dy
    pts_z = origin[2] + np.array(zlocs)
    if ct_dims.flipped:
        pts_z = pts_z[..., ::-1]
    return [pts_x, pts_y, pts_z]


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
    from_dose: bool = False

    def __post_init__(self):
        if 'RTDOSE' in self.dicoms and self.from_dose:
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
        self.position = ds.PatientPosition
        self.dicoms = None

    def _calculate_dz(self, zlocations):
        zlocations.sort()
        differences = np.zeros((len(zlocations) - 1))
        previous = min(zlocations)

        for i in range(1, len(zlocations)):
            differences[i - 1] = round(abs(previous - zlocations[i]), 2)
            previous = zlocations[i]

        differences = list(set(differences))

        if len(differences) > 1:
            self.multi_thick = True
        return min(differences)

    def _calc_n_slices(self, files: list):
        """Calculates the number of volume slices

        Args:
            files (DicomFile): A list of DicomFile objects

        Notes:
            Creating the volume by the difference in slice location at high and
            low instances ensures proper registration to rstructs, even if
            images slices are missing. We can interpolate to the lowest
            instance if we do not have instance 1, but we cannot extrapolate
            if higher instances are missing
        """
        z0, z1 = (np.inf, -np.inf)
        low_inst = np.inf
        low_thickness = None
        header_thicknesses = []
        zlocations = []

        for dcm in files:
            ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
            self.ipp = ds.ImagePositionPatient
            self.iop = ds.ImageOrientationPatient
            header_thicknesses.append(float(ds.SliceThickness))
            zloc = float(self.ipp[-1])
            zlocations.append(zloc)

            if zloc < z0:
                z0 = zloc
            if zloc > z1:
                z1 = zloc

            if hasattr(ds, 'InstanceNumber'):
                inst = int(ds.InstanceNumber)
                if inst < low_inst:
                    low_inst = inst
                    low_thickness = ds.SliceThickness

        if 1 < low_inst < 5 and low_inst != np.inf:  # For extrapolation of missing slices
            # Need to make it the slice thickness of the lowest slice
            # in case the image has mixed thicknesses
            z0 -= low_thickness * (low_inst - 1)
            low_inst = 1

        self.dz = self._calculate_dz(zlocations)
        self.zlohi = (z0, z1)
        self.origin = np.array([*self.ipp[:2], max(z0, z1)])
        self.slices = 1 + round((max(z0, z1) - min(z0, z1)) / self.dz)

        return ds

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
    """Converts a dictionary into a dataclass

    Args:
        d (dict): Dict of name and properties for the dataclass
        name (str, optional): Name of dataclass. Defaults to 'd_dataclass'.

    Returns:
        dataclass object: Created dataclass
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


def flatten_list(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten_list(S[0]) + flatten_list(S[1:])
    return S[:1] + flatten_list(S[1:])


def split_tree(primary: NodeMixin, n: int = 10) -> list:
    """Splits a tree into a series of n-sized trees

    Args:
        primary (NodeMixin): Primary tree to split
        n (int, optional): Size of each smaller tree. Defaults to 10.

    Returns:
        list: A list of the split trees
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
    """Combine a series of trees into the primary tree

    Args:
        primary (NodeMixin): Primary tree to return with secondaries
        secondaries (list): Secondaries to add to primary tree

    Returns:
        NodeMixin: Unified tree under type Primary
    """
    for tree in secondaries:
        for child in tree:
            primary.adopt(child)
    return primary


def insert_into_tree(tree: NodeMixin, mod_ptr_pairs: list) -> None:
    """Inserts a volume file into a tree

    Args:
        tree (NodeMixin): Tree to insert volume files
        mod_ptr_pairs (list): list of modality and corresponding file
    """
    for modality, leaf in mod_ptr_pairs:
        node = tree
        for a in modality.ancestors:
            node = anytree.search.findall(node, filter_=lambda x: x.name == a.name)[-1]
        node._add_file(leaf)


class ReconToPath:
    """For reconstruction of a tree

    Args:
        tree (NodeMixin): Tree to reconstruct
        path (str): Path to save reconstructed volume

    Returns:
        list: A list of tuples containing modality and ReconstructedFile
    """
    def __init__(self, path: str = None):
        self.path = path

    def __call__(self, tree: NodeMixin):
        return tree._recon_to_disk(return_mods=True, path=self.path)


def threaded_recon(primary: NodeMixin, path: str) -> NodeMixin:
    """A multiprocessed reconstruction of primary

    Args:
        primary (NodeMixin): Tree for reconstruction

    Returns:
        NodeMixin: Primary with the volume files inserted

    Notes:
        Reconstruction does not scale with processors, so we split the
            tree into smaller trees, reconstruct each in parallel and
            then recombine the trees
    """
    trees = split_tree(primary, n=10)

    ncpus = multiprocessing.cpu_count()
    recon_fn = ReconToPath(path)

    with ProcessPool(max_workers=ncpus//4) as P:
        results = list(P.map(recon_fn, trees))
        P.shutdown()
    ProcessPool().shutdown()

    primary = combine_trees(primary, trees)

    for mod_ptr_pairs in results:
        insert_into_tree(primary, mod_ptr_pairs)

    return primary


def decendant_types(group: NodeMixin) -> list:
    """Declare the decendant types for a group, could
       also use group.decendant

    Args:
        group (NodeMixin): The group to determine decendants

    Returns:
        list: List of decendant types
    """
    heiarchy = ['Cohort', 'Patient', 'FrameOfRef', 'Study', 'Series', 'Modality']
    index = heiarchy.index(group)
    return heiarchy[index:]


def structure_voxel_count(tree: NodeMixin, structure: str) -> dict:
    """The count of voxels in each occurance of a structure

    Args:
        tree (NodeMixin): The tree to iterate through
        structure (str): String corresponding to structure name

    Returns:
        dict: Dict of SeriesInstanceUID and counts
    """
    it = tree.iter_struct_volume_files()
    counts = {}
    for volfile in it:
        if volfile.Modality == 'RTSTRUCT':
            original_ptr = volfile.is_file()
            if original_ptr:
                volfile.load_array()
            for name, volume in volfile.volumes.items():
                if name == structure:
                    counts.update({volfile.name: np.sum(volume)})
            if original_ptr:
                volfile.convert_to_file()
    return counts


def clean_up(arr: np.ndarray) -> np.ndarray:
    """Clean an array by removing any discontinuities

    Args:
        arr (np.ndarray): boolean array to be cleaned

    Returns:
        np.ndarray: cleaned boolean array
    """
    encoded, n_labels = measure.label(arr, connectivity=3, return_num=True)
    if n_labels >= 2:
        value, count = np.unique(encoded, return_counts=True)
        encoded[encoded != value[np.argmax(count[1:]) + 1]] = 0
        arr = arr * encoded
    return np.array(arr, dtype='bool')


def fill_holes(arr: np.ndarray, axially: bool = True) -> np.ndarray:
    """Fills holes either volumetrically or axially (default)

    Args:
        arr (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    if not axially:
        return ndimage.binary_fill_holes(arr)
    for z in range(arr.shape[-1]):
        arr[..., z] = ndimage.binary_fill_holes(arr[..., z])
    return arr


def smooth(arr: np.ndarray, iterations: int = 2) -> np.ndarray:
    big = ndimage.binary_dilation(arr, iterations=iterations)
    return ndimage.binary_erosion(big, iterations=iterations)


def smooth_median(arr: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    return ndimage.median_filter(arr, kernel_size)


def expand_contour(arr: np.ndarray, distance: float, voxel_size: list,
                   epsilon: float = 0) -> np.ndarray:
    """Expand contour uniformly in all dimensions by a given distance

    Args:
        arr (np.ndarray): Boolean numpy array to expand
        distance (float): Distance, in voxel_size dimension units to expand
        voxel_size (list): Voxel dimensions in either pixels [1, 1, 1] or mm
        epsilon (float, optional): Epsilon to offset the expansion, same
            as adding to the distance. Potentially helpful because software, like
            MIM, contours at voxel_size[0]/2 dimensions an to match a MIM expansion may warrant
            an offset of voxel_size[0]/2. Defaults to 0.

    Returns:
        np.ndarray: [description]
    """
    padded_distance = distance + epsilon
    surface = ndimage.binary_erosion(arr) ^ np.array(arr, dtype=np.bool)
    inverted = np.invert(surface)
    edt = ndimage.distance_transform_edt(inverted, sampling=voxel_size)
    return edt < padded_distance


def dose_max_points(dose_array: np.ndarray,
                    dose_coords: np.ndarray = None) -> np.ndarray:
    """Calculates the dose maximum point in an array, returns index or coordinates

    Args:
        dose_array (np.ndarray): A reconstructed dose array
        dose_coords (np.ndarray, optional): Associated patient coordinates. Defaults to None.

    Returns:
        np.ndarray: The dose max index, or patient coordinates, if given
    """
    index = np.unravel_index(np.argmax(dose_array), dose_array.shape)

    if dose_coords:
        return dose_coords[index]
    return index
