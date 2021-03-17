import anytree
from anytree import Node, RenderTree
from anytree import NodeMixin
from anytree.iterators.levelorderiter import LevelOrderIter
from anytree.iterators.levelordergroupiter import LevelOrderGroupIter
from dataclasses import dataclass
import pydicom
from glob import glob
import os
from processing import Reconstruction, Deconstruction
import warnings
import shutil
from datetime import datetime
from typing import Any, TypeVar


# TODO
# - Make file saving date option functional
# - Change path declarations to pathlib objects


# Move to utils.py
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


def save_tree(tree: NodeMixin, path: str, prefix: str = 'group') -> None:
    """[saves copy of dicom files to specified location, ordered
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


class GroupUtils(NodeMixin):
    """[General utilities for all groups]

    Args:
        NodeMixin ([NodeMixin]): [AnyTree node mixin]
        name (str, optional): [name of the group]. Defaults to None.
        files (list, optional): [list of absolute paths
            to *.dcm files]. Defaults to None.
        parent (Node, optional): [the parent Node]. Defaults to None.
        children (tuple, optional): [a tuple of Nodes which are the
            children of the current Node]. Defaults to None.
        include_series (bool, optional): [specifies if FrameOfRef points to
            Series (True) or Modality (False)]. Defaults to False.

    Methods:
        merge (NodeMixin): merges another group into current
        steal (NodeMixin): steals children from another group
        append (NodeMixin): appends children to group
        prune (str): prunes specified branch from tree
        flatten (None): flattens tree to one child per parent
    """
    def __init__(self, name=None, files=None, parent=None, children=None,
                 include_series=False):
        super().__init__()
        self.name = name
        self.files = files
        self.parent = parent
        self.include_series = include_series
        if children:
            self.children = children

    def __iter__(self):
        return iter(self.children)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        output = []
        for pre, fill, node in RenderTree(self, childiter=self._str_sort):
            line = f'{pre}{node.__class__.__name__}: {node.name}'
            if node.__class__.__name__ == 'Modality':
                line += node.__str__()
            output.append(line)
        return '\n'.join(output)

    def _str_sort(self, items: list) -> list:
        return sorted(items, key=lambda item: item.name)

    def _digest(self):
        """[digests the file paths, building the tree]
        """
        while self.files:
            f = self.files.pop()
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            self._add_file(f)

    def _add_file(self, dicomfile: object) -> None:
        """[adds a file to the file tree]

        Args:
            dicomfile (DicomFile): [a DicomFile object]
        """
        key = str(dicomfile[self._organize_by])
        found = False
        for child in self.children:
            if key == child.name:
                child._add_file(dicomfile)
                found = True
        if not found:
            dt = dicomfile.DateTime
            print(dt)
            child_params = {'name': key,
                            'files': [dicomfile],
                            'parent': self,
                            'include_series': self.include_series}
            self._child_type(**child_params)

    @property
    def dirname(self):
        if hasattr(self, '_dirname'):
            return self._dirname
        return repr(self) + '_' + self.name

    @dirname.setter
    def dirname(self, name: str):
        self._dirname = name

    @property
    def datename(self):
        if hasattr(self, '_datename'):
            return self._datename
        return self.DateTime + '_' + self.name

    @datename.setter
    def datename(self, name):
        self._datename = name

    def merge(self, other: NodeMixin) -> None:
        """[merges two groups]

        Args:
            other (NodeMixin): [merged into primary group]
        """
        other.parent = self.parent

    def steal(self, other: NodeMixin) -> None:
        """[steals children from one parent]

        Args:
            other (NodeMixin): [parent who loses children]

        Notes:
            Porter et al. do not condone kidnapping
        """
        self.children += other.children

    def append(self, others: NodeMixin) -> None:
        """[appends children onto parent]

        Args:
            others (NodeMixin): [children to be appended]
        """
        for other in others:
            other.parent = self

    def prune(self, childname: str) -> None:
        """[prunes branch from tree]

        Args:
            childname (str): [name of branch to prune]
        """
        for child in self.children:
            if child.name in childname:
                child.parent = None

    def flatten(self) -> None:
        """[flatten results in each parent having one child, except for
            the Modality group. Flattened with specified group as root]
        """
        if self.include_series:
            limit = 'Series'
        else:
            limit = 'FrameOfRef'
        if repr(self) != limit:
            all_children = LevelOrderGroupIter(self, maxlevel=2)
            _ = next(all_children)  # first item is self
            for group in all_children:
                for i, child in enumerate(group):
                    child.flatten()
                    if i > 0:
                        group[0].steal(child)
                        child.parent.prune(child.name)


class DicomFile(GroupUtils):
    """[Group level for individal Dicom files, pulls releveant header data out]

    Args:
        filepath (str): absolute file path to the *.dcm file
    """
    SelfType = TypeVar('DicomFile')

    def __init__(self, filepath: str, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.name = os.path.basename(filepath)
        self.filepath = filepath
        self._digest()

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __lt__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation < other.SliceLocation
        return self.InstanceUID < other.InstanceUID

    def __gt__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation > other.SliceLocation
        return self.FrameOfRefUID > other.FrameOfRefUID

    def __eq__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation == other.SliceLocation
        return self.FrameOfRefUID == other.FrameOfRefUID

    def _digest(self):
        with pydicom.dcmread(self.filepath, stop_before_pixels=True) as ds:
            self.PatientID = ds.PatientID
            self.StudyUID = ds.StudyInstanceUID
            self.SeriesUID = ds.SeriesInstanceUID
            self.Modality = ds.Modality
            self.InstanceUID = ds.SOPInstanceUID
            self.DateTime = DicomDateTime(ds)

            if hasattr(ds, 'FrameOfReferenceUID'):
                self.FrameOfRefUID = ds.FrameOfReferenceUID
            elif hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
                ref_seq = ds.ReferencedFrameOfReferenceSequence
                self.FrameOfRefUID = ref_seq[0].FrameOfReferenceUID
            else:
                self.FrameOfRefUID = None

            if hasattr(ds, 'Columns'):
                self.cols = ds.Columns
                self.rows = ds.Rows

            if hasattr(ds, 'SliceLocation'):
                self.SliceLocation = float(ds.SliceLocation)
            elif hasattr(ds, 'ImagePositionPatient'):
                self.SliceLocation = float(ds.ImagePositionPatient[-1])


class Cohort(GroupUtils):
    """[Group level for Cohort]

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): [specifies if FrameOfRef points to
            Series (True) or Modality (False)]. Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Patient
        self._organize_by = 'PatientID'
        self._digest()


class Patient(GroupUtils):
    """[Group level for Patient, specified by PatientUID]

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): [specifies if FrameOfRef points to
            Series (True) or Modality (False)]. Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Study
        self._organize_by = 'StudyUID'
        self._digest()


class Study(GroupUtils):
    """[Group level for Study, specified by StudyUID]

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): [specifies if FrameOfRef points to
            Series (True) or Modality (False)]. Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = FrameOfRef
        self._organize_by = 'FrameOfRefUID'
        self._digest()


class FrameOfRef(GroupUtils):
    """[Group level for FrameOfReference, specified by FrameOfReferenceUID]

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): [specifies if FrameOfRef points to
            Series (True) or Modality (False)]. Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        if self.include_series:
            self._child_type = Series
            self._organize_by = 'SeriesUID'
        else:
            self._child_type = Modality
            self._organize_by = 'Modality'
        self._digest()


class Series(GroupUtils):
    """[Group level for Series, specified by SeriesUID]

    Args:
        name (str): A string declaring the group name
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Modality
        self._organize_by = 'Modality'
        self._digest()


class Modality(GroupUtils):
    """[Group level for Modality, specified by Modality]

    Args:
        name (str): A string declaring the group name

    Attributes:
        ct: [ct files within group]
        nm: [nm files within group]
        mr: [mr files within group]
        pet: [pet files within group]
        dose: [dose files within group]
        struct: [structf files within group]
    """
    ct = property(mod_getter('CT'))
    nm = property(mod_getter('NM'))
    mr = property(mod_getter('MR'))
    pet = property(mod_getter('PET'))
    dose = property(mod_getter('RTDOSE'))
    struct = property(mod_getter('RTSTRUCT'))

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dirname = name
        self.data = {}
        self.recon = Reconstruction()
        self.decon = Deconstruction()
        self._child_type = DicomFile
        self._organize_by = 'Modality'
        self._digest()

    def __str__(self):
        middle = []
        for index, key in enumerate(self.data):
            middle.append(f' {len(self.data[key])} files')
        output = ' [' + ','.join(middle) + ' ]'
        return output

    def _add_file(self, dicomfile: DicomFile):
        key = str(dicomfile[self._organize_by])
        if key in self.data.keys():
            self.data[key].append(dicomfile)
        else:
            self.data.update({key: [dicomfile]})


@dataclass
class DicomDateTime:
    """[pulls the relevant date and time info from the DICOM header]

    Attributes:
        StudyDate: [(0008,0020) DICOM Tag]
        SeriesDate: [(0008, 0021) DICOM Tag]
        ContentDate: [(0008, 0023) DICOM Tag]
        AcquisitionDate: [(0008, 0022) DICOM Tag]
        StudyTime: [(0008,0030) DICOM Tag]
        SeriesTime: [(0008, 0031) DICOM Tag]
        ContentTime: [(0008, 0033) DICOM Tag]
        AcquisitionTime: [(0008, 0032) DICOM Tag]

    Methods:
        iso_format (str): [returns ISO 8601 format date time for group type]
    """
    ds: pydicom.dataset

    def __post_init__(self):
        prefix = ['Study', 'Series', 'Acquisition', 'Content',
                  'InstanceCreation', 'StructureSet']
        suffix = ['Date', 'Time']
        attrs = [p + s for p in prefix for s in suffix]
        for name in attrs:
            if hasattr(self.ds, name):
                value = getattr(self.ds, name)
                setattr(self, name, value)
            else:
                setattr(self, name, None)
        self.modality = self.ds.Modality
        #self.ds = None

    def __setitem__(self, name: str, value: Any):
        self.__dict__[name] = value

    def __str__(self):
        return self.iso_format('Study')

    def iso_format(self, group: str):
        """[returns ISO 8601 format date time for group type]

        Args:
            group (str): [string representing group type]

        Returns:
            [str]: [ISO 8601 formatted date time string with milliseconds]
        """
        # ISO 8601 formatted date time string
        if group == 'Study' and self.StudyDate is not None:
            date, time = (self.StudyDate, self.StudyTime)
        elif group == 'Series' and self.SeriesDate is not None:
            date, time = (self.SeriesDate, self.SeriesTime)
        elif self.InstanceCreationDate is not None:
            date, time = (self.InstanceCreationDate, self.InstanceCreationTime)
        elif self.AcquisitionDate is not None:
            date, time = (self.AcquisitionDate, self.AcquisitionTime)
        elif self.ContentDate is not None:
            date, time = (self.ContentDate, self.ContentTime)
        else:
            date, time = ('19700101', '000000.000000')

        #del self.ds[(0x3006, 0x0080)]
        #del self.ds[(0x3006, 0x0010)]
        #del self.ds[(0x3006, 0x0020)]
        #del self.ds[(0x3006, 0x0039)]
        #print(self.ds)

        dateobj = datetime.strptime(date+time, '%Y%m%d%H%M%S.%f')
        dateobj.replace(microsecond=0)
        str_name = dateobj.isoformat()
        str_name.replace(':', '.')
        print(str_name)
        return str_name



if __name__ == '__main__':
    files = glob('/home/eporter/eporter_data/hippo_data/**/*.dcm', recursive=True)

    cohort = Cohort(name='TestFileSave', files=files, include_series=False)

    for patient in cohort:
        patient.flatten()
        #print(patient.datename)

    #print(cohort)
    #save_tree(cohort, '/home/eporter/eporter_data/', prefix=None)
