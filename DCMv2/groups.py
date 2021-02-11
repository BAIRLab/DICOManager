import pydicom
from glob import glob
from copy import deepcopy
from processing import Reconstruction, Deconstruction
from dataclasses import dataclass, fields, field
from fileutils import FileUtils, DicomGroup, DicomFile
from datetime import datetime


@dataclass
class BasicData:
    path: str = None
    all_files: list = None
    name: str = None
    data: dict = None
    filter_list: list = None
    _identifer: str = None
    _files: DicomGroup = None
    _tot_string: str = ''

@dataclass
class Modality(FileUtils):
    # A single Modality group would be for, say MR
    # Within that would be a dictionary with each key being
    # the frame of reference and the value being a DicomGroup
    # for that given image set
    modality_type: str
    _modality_type: str = field(init=False)
    data: dict = None
    organize_by: str = 'FrameOfReference'
    immutable_modality: bool = True

    def group_id(self):
        if hasattr(self, 'FrameOfReference'):
            return self.FrameOfReference
        return None

    @property
    def modality_type(self):
        self._modality_type

    @modality_type.setter
    def modality_type(self, modality_type):
        if not hasattr(self, '_modality_type'):
            self._modality_type = modality_type
        elif not self.immutable_modality:
            self._modality_type = modality_type
        else:
            message = 'Modality.immutable_modality=True by default'
            raise TypeError(message, UserWarning)


@dataclass
class NewSeries(FileUtils):
    data: dict = None  # That is comprised of modality groups: CT0 ... CT#, MR0 ... MR#
    files: list = None
    path: str = None
    recon: object = Reconstruction()
    decon: object = Deconstruction()
    child_type: object = Modality

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'SeriesUID'):
            return self.SeriesUID
        return None

    def frame_of_ref_subset(self, frameUID):
        FoRSeries = NewSeries()

        for mod, group in self.data.items():
            if group == frameUID:
                FoRSeries.add(group)
        return NewSeries


@dataclass
class Series(FileUtils, BasicData):
    ct: dict = None
    nm: dict = None
    mr: dict = None
    pet: dict = None
    struct: dict = None
    dose: dict = None
    recon: object = Reconstruction()
    decon: object = Deconstruction()
    #sorter: object = Sort.by_series()
    _organize_by: str = 'Modality'
    _child_type: object = DicomGroup
    _hierarchy: int = 4

    def __post_init__(self):
        self.digest_data()

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def __iter__(self):
        modalities = [x.name for x in fields(self)]
        return iter(modalities[:6])

    def __repr__(self):
        for mod in self:
            print(mod, ' : ', len(self[mod]))

    def group_id(self):
        if hasattr(self, 'SeriesUID'):
            return self.SeriesUID

    def frame_of_ref_subset(self, frameUID):
        NewSeries = Series()

        for mod in self:
            for dlist in mod:
                if dlist.FrameOfRef == frameUID:
                    NewSeries.append(dlist)

        return NewSeries


@dataclass
class Study(FileUtils, BasicData):
    _organize_by: str = 'SeriesUID'
    _child_type: object = Series
    _hierarchy: int = 3

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'StudyUID'):
            return self.StudyUID
        return None


@dataclass
class Patient(FileUtils, BasicData):
    # Frame of References associated with a data group
    # Representing a patient with N study groups
    _organize_by: str = 'StudyUID'
    _child_type: object = Study
    _hierarchy: int = 2

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'PatientID'):
            return self.PatientID
        return None



@dataclass
class Cohort(FileUtils, BasicData):
    name: str = None
    _organize_by: str = 'PatientID'  # specifies how this group is sorted and filtered
    _child_type: object = Patient
    _hierarchy: int = 1

    def __post_init__(self):
        datetime_str = datetime.now().strftime('%H%M%S')
        if self.name is None:
            self._identifer = 'Cohort_from_' + datetime_str
        else:
            self._identifer = self.name

        self.digest_data()

    def group_id(self):
        if hasattr(self, 'identifier'):
            return self.identifier
        return None
