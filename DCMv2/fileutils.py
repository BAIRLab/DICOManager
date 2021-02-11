import pydicom
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from glob import glob
from abc import ABC, abstractmethod

# If we populate this with the __eq__ parameter as SliceLocation,
# we can use that as a means of sorting the dataset


@dataclass
class DicomFile:
    filepath: str

    def __post_init__(self):
        ds = pydicom.dcmread(self.filepath, stop_before_pixels=True)
        self.PatientID = ds.PatientID
        self.StudyUID = ds.StudyUID
        self.SeriesUID = ds.SeriesUID
        self.Modality = ds.Modality
        self.FrameOfRefUID = ds.FrameOfReferenceUID
        self.InstanceUID = ds.InstanceUID
        self.cols = ds.Columns
        self.rows = ds.Rows
        if hasattr(ds, 'SliceLocation'):
            self.SliceLocation = ds.SliceLocation

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


# We can make this no initialized with @dataclass(init=False)
@dataclass
class DicomGroup:
    filepaths: list = None  # input posix paths
    filedata_list: list = None  # list of FileData objects
    slice_range: [np.inf, -np.inf]

    def __post_init__(self):
        if self.filepaths is not None:
            for f in self.filepaths:
                self.append(f)

    def __getitem__(self, name):
        return getattr(self, name)

    def __iter__(self):
        return iter(self.filedata_list)

    def __lt__(self, other):
        return self.FrameOfRefUID < other.FrameOfRefUID

    def __gt__(self, other):
        return self.FrameOfRefUID > other.FrameOfRefUID

    def __eq__(self, other):
        return self.FrameOfRefUID == other.FrameOfRefUID

    def __len__(self):
        return len(self.filedata_list)

    def uid_check(self, dicomfile: DicomFile):
        patient = dicomfile.PatientID == self.PatientID
        study = dicomfile.StudyUID == self.StudyUID
        series = dicomfile.SeriesUID == self.SeriesUID
        modality = dicomfile.Modality == self.Modality
        ref_frame = dicomfile.FrameOfRefUID == self.FrameOfRefUID

        return all([patient, study, series, modality, ref_frame])

    def append(self, filename: DicomFile):
        if filename.__class__ is not DicomFile:
            filename = DicomFile(filename)

        if self.uid_check(filename):
            loc = filename.SliceLocation
            if loc < self.SliceRange[0]:
                self.SliceRange[0] = loc
            if loc > self.SliceRange[0]:
                self.SliceRange[1] = loc
            self.filedata_list.append(filename)


# Make into an abstractclass because this is never instantiated
# Instead, this is generally inherited by all groups
class FileUtils(ABC):
    # A class to sort files and basic group management
    def __iter__(self):
        return self.data.values()

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value):
        self.data[name] = value

    def __repr__(self):
        indent = 0
        if self.group_id is not None:
            indent += 1
            print(self.group_id)
        for attr, value in self.data.items():
            print('\t' * indent, attr)
            print('\t' * (indent + 1), value)  # value is __repr__ of lower group

    def add(self, object):
        # Add to higher order
        value = object.group_id
        if value not in self.data:
            self.data[value] = object
        else:
            self.data[value].append(object)
        return None

    def append(self, object):
        # Append to higher order
        message = "Merge is for same group type, append is for different"
        assert issubclass(self.__class__, object.__class__), message
        new_object = self.update(object)
        self.add(new_object)

    def pop(self, uid):
        # Remove item from data dictionary
        if uid in self.data:
            return self.data.pop(uid)
        return None

    def prune(self, uid):
        _ = self.pop(uid)

    def merge(self, object):
        # Merge two of equal order
        message = "Merge is for same group type, append is for different"
        assert self.__class__ is object.__class__, message
        new_object = self.update(object)
        new_dict = {**self.data, **new_object.data}
        self.data = new_dict

    def update(self, object):
        # This changes the UIDs necessary to make it into the new grouping
        new_object = deepcopy(object)
        new_object.FileData.group_id = self.group_id
        return new_object

    def digest_data(self, name):
        message = "Can only specify a path of list of files"
        assert self.path is not None and self.files is not None, message

        if self.path is not None:
            self.all_files = glob(self.path + '/**/*.dcm', recursive=True)

        if self.filter_list is not None:
            self.all_files = self.file_filter()

        self.sort()

        if self[name] is None:
            f = self.all_files[0]
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            self[name] = f.__dataclass_fields__[name]

    def sort(self):
        if self.all_files.__class__ is not DicomGroup:
            self.all_files = DicomGroup(self.all_files)

        fill_dict = {}
        files = deepcopy(self.all_files)

        while files:
            f = files.pop()
            if f[self.organzie_by] not in fill_dict:
                fill_dict[f[self.organize_by]] = DicomGroup(f)
            else:
                fill_dict[f[self.organize_by]].append(f)

        for key, data in fill_dict.items():
            params = {'all_files': data,
                      self.organize_by: key}
            self.data[key] = self.child_type(**params)

    def file_filter(self):
        filtered_files = []
        files = deepcopy(self.all_files)
        while files:
            f = files.pop()
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            if f[self.organize_by] in self.filter_list:
                filtered_files.append(f)
        return filtered_files

    @abstractmethod
    def group_id(self):
        pass
