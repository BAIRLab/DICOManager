import pydicom
from glob import glob
from copy import deepcopy
from proessing import Reconstruction, Deconstruction
from dataclasses import dataclass, fields, field
from fileutils import FileUtils, DicomGroup, DicomFile


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
    data: dict = None # That is comprised of modality groups: CT0 ... CT#, MR0 ... MR#
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
class Series(FileUtils):
    ct: dict = None
    nm: dict = None
    mr: dict = None
    pet: dict = None
    struct: dict = None
    dose: dict = None
    files: list = None
    path: str = None
    recon: object = Reconstruction()
    decon: object = Deconstruction()
    #sorter: object = Sort.by_series()
    child_type: object = DicomGroup

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
class Study(FileUtils):
    # Representing an image study group
    # Can only do reconstruction on this type
    # This group can also be manually filled
    path: str = None
    files: list = None  # All files contained withing group
    data: dict = None
    organize_by: str = 'SeriesUID'
    child_by: object = Series
    #sorter: object = Sort.by_study

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'StudyUID'):
            return self.StudyUID
        return None


@dataclass
class Patient(FileUtils):
    # Frame of References associated with a data group
    # Representing a patient with N study groups
    all_files: list = None  # All files contained within group
    path: str = None  # A path to the directory
    PatientID: str = None
    data: dict = None
    organize_by: str = 'StudyUID'
    child_type: object = Study
    #sorter: object = Sort.by_study

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'PatientID'):
            return self.PatientID
        return None


@dataclass
class Cohort(FileUtils):  # Alternatively 'Dataset'
    # A data set of multiple patients
    path: str
    all_files: list = None
    name: str = None
    data: dict = None
    filter_list: list = None
    organize_by: str = 'PatientID'  # specifies how this group is sorted and filtered
    child_type: object = Patient
    #sorter: object = Sort.by_patient
    #filterer: object = Filter.by_patient

    def __post_init__(self):
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'name'):
            return self.name
        return None
