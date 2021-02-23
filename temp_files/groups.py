import pydicom
from glob import glob
from copy import deepcopy
from processing import Reconstruction, Deconstruction
from dataclasses import dataclass, fields, field
from fileutils import FileUtils, DicomGroup, DicomFile, immutable
from datetime import datetime


@dataclass
class BasicData:
    path: str = None
    all_files: list = None
    data: dict = None
    filter_list: list = None
    _identifier: str = None
    _files: DicomGroup = None
    _tot_string: str = ''


@dataclass
class Modality(FileUtils, BasicData):
    # A single Modality group would be for, say MR
    # Within that would be a dictionary with each key being
    # the frame of reference and the value being a DicomGroup
    # for that given image set
    _organize_by: str = 'FrameOfReference'
    _child_type: object = DicomGroup
    _depth: int = 5

    def __post_int__(self):
        print('in modalities')
        self.digest_data()

    def group_id(self):
        if hasattr(self, 'FrameOfReference'):
            return self.FrameOfReference
        return None

@dataclass
class NewSeries(FileUtils, BasicData):
    recon: object = Reconstruction()
    decon: object = Deconstruction()
    _organize_by: str = 'Modality'
    _child_type: object = Modality
    _depth: int = 4

    def __post_init__(self):
        self.digest_data()

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
    _organize_by: str = 'Modality'
    _child_type: object = DicomGroup
    _depth: int = 4

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

    def frame_of_ref_subset(self, frameUID):
        NewSeries = Series()

        for mod in self:
            for group in mod:
                if group.FrameOfRef == frameUID:
                    NewSeries.append(group)

        return NewSeries


@dataclass
class Study(FileUtils, BasicData):
    _organize_by: str = 'SeriesUID'
    _child_type: object = NewSeries
    _depth: int = 3

    def __post_init__(self):
        self.digest_data()


@dataclass
class Patient(FileUtils, BasicData):
    # Frame of References associated with a data group
    # Representing a patient with N study groups
    _organize_by: str = 'StudyUID'
    _child_type: object = Study
    _depth: int = 2

    def __post_init__(self):
        self.digest_data()


@dataclass
class Cohort(FileUtils, BasicData):
    _organize_by: str = 'PatientID'  # specifies how this group is sorted and filtered
    _child_type: object = Patient
    _depth: int = 1

    def __post_init__(self):
        datetime_str = datetime.now().strftime('%H%M%S')
        if self._identifier is None:
            self._identifier = 'Cohort_from_' + datetime_str

        self.digest_data()
