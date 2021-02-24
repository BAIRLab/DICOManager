import anytree
from anytree import Node, RenderTree
from anytree import NodeMixin
from anytree.iterators.levelordergroupiter import LevelOrderGroupIter
from dataclasses import dataclass
import pydicom
from glob import glob
import os
from processing import Reconstruction, Deconstruction
import warnings


class GroupUtils(NodeMixin):
    def __init__(self, name=None, files=None, parent=None, children=None,
                 include_series=False):
        super().__init__()
        self.name = name
        self.files = files
        self.parent = parent
        self.include_series = include_series
        if children:
            self.children = children

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

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

    def __iter__(self):
        return iter(self.children)

    def _str_sort(self, items):
        return sorted(items, key=lambda item: item.name)

    def _digest(self):
        while self.files:
            f = self.files.pop()
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            self._add_file(f)

    def _add_file(self, dicomfile):
        key = str(dicomfile[self._organize_by])
        found = False
        for child in self.children:
            if key == child.name:
                child._add_file(dicomfile)
                found = True
        if not found:
            child_params = {'name': key,
                            'files': [dicomfile],
                            'parent': self,
                            'include_series': self.include_series}
            _ = self._child_type(**child_params)

    def merge(self, other):
        # merges two parents
        other.parent = self.parent

    def steal(self, other):
        # steals the children from another parent
        self.children += other.children

    def append(self, others):
        # appends children to another parent
        for other in others:
            other.parent = self

    def prune(self, childname):
        # prunes branches from the tree
        for child in self.children:
            if child.name in childname:
                child.parent = None

    def flatten(self):
        # ensures each parent has one child (except for modality)
        # Warning: currently does not change dicom headers
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


class Cohort(GroupUtils):
    # For cohort and below, we need to take series as a variable
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Patient
        self._organize_by = 'PatientID'
        self._digest()


class Patient(GroupUtils):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Study
        self._organize_by = 'StudyUID'
        self._digest()


class Study(GroupUtils):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = FrameOfRef
        self._organize_by = 'FrameOfRefUID'
        self._digest()


class FrameOfRef(GroupUtils):
    # A series must share a FrameOfReference, therefore we can
    # exclude series for the default sorting
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
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Modality
        self._organize_by = 'Modality'
        self._digest()


def mod_getter(modtype):
    def func(self):
        modlist = []
        if modtype in self.data:
            modlist.append(self.data[modtype])
        return modlist
    return func


class Modality(GroupUtils):
    ct = property(mod_getter('CT'))
    nm = property(mod_getter('NM'))
    mr = property(mod_getter('MR'))
    pet = property(mod_getter('PET'))
    dose = property(mod_getter('RTDOSE'))
    struct = property(mod_getter('RTSTRUCT'))

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
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

    def _add_file(self, dicomfile):
        key = str(dicomfile[self._organize_by])
        if key in self.data.keys():
            self.data[key].append(dicomfile)
        else:
            self.data.update({key: [dicomfile]})

class DicomFile(GroupUtils):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.name = os.path.basename(filename)
        self._filename = filename
        self._digest()

    def _digest(self):
        ds = pydicom.dcmread(self._filename)
        self.PatientID = ds.PatientID
        self.StudyUID = ds.StudyInstanceUID
        self.SeriesUID = ds.SeriesInstanceUID
        self.Modality = ds.Modality
        self.InstanceUID = ds.SOPInstanceUID

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

    def __getitem__(self, name):
        return getattr(self, name)

    def __lt__(self, other):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation < other.SliceLocation
        return self.InstanceUID < other.InstanceUID

    def __gt__(self, other):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation > other.SliceLocation
        return self.FrameOfRefUID > other.FrameOfRefUID

    def __eq__(self, other):
        if hasattr(self, 'SliceLocation') and hasattr(self, 'SliceLocation'):
            return self.SliceLocation == other.SliceLocation
        return self.FrameOfRefUID == other.FrameOfRefUID


if __name__ == '__main__':
    files = glob('/home/eporter/eporter_data/hippo_data/**/*.dcm', recursive=True)
    cohort = Cohort(name='Cohort1', files=files[:], include_series=False)

    #for patient in cohort:
    #    patient.flatten()

    print(cohort)
