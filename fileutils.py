import pydicom
import numpy as np
from dataclasses import dataclass, is_dataclass
from copy import deepcopy
from glob import glob
from abc import ABC, abstractmethod
import warnings


# We may want to prevent users from changing the group metadata to not represent
# the files contained within the group. We could either constantly read the files
# to check they haven't been tampered with, let the user shoot themselves in the
# foot or default to having protected attributes unless otherwise specified. Thus,
# we could restrict the functionality to higher order user facing functions
def immutable(imm_fields):
    """[a decorator to make specified fields 'immutable'. This prevents
        the specified fields from being altered unless <class>.immutable=False]

    Args:
        imm_field ([list OR str]): [field name(s) to make immutable]

    Raises:
        TypeError: [if immutable_field is not str or list]

    Warns:
        UserWarning: [if attempting to mutate immutable variable]

    Returns:
        [Object]: [returns inherited class of decorated type]
    """
    if not isinstance(imm_fields, list):
        if isinstance(imm_fields, str):
            imm_fields = [imm_fields]
        else:
            raise TypeError('Input must be a field name or list of names')
    assert 'immutable' not in imm_fields, 'attribute immutable is protected'

    def decorator(OldClass):
        class NewClass(OldClass):
            """[A new class with immutable fields]

            Args:
                OldClass ([object]): [decorated type class]
            """

            def __init__(self, *args, **kwargs):
                if not is_dataclass(OldClass):
                    # If type is not dataclass, it must be initialized
                    super().__init__(*args, **kwargs)

                if not hasattr(self, 'immutable'):
                    self.immutable = True

            def __setattr__(self, key, value):
                # set field if not immutable type, otherwise _set_attr
                if key in imm_fields and hasattr(self, key) and self.immutable:
                    message = '<class>.immutable=True, change to mutate'
                    warnings.warn(message, UserWarning)
                else:
                    object.__setattr__(self, key, value)

        return NewClass
    return decorator


@dataclass
class DicomFile:
    filepath: str

    def __post_init__(self):
        ds = pydicom.dcmread(self.filepath, stop_before_pixels=True)
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


# We can make this no initialized with @dataclass(init=False)
@dataclass
class DicomGroup:
    filepaths: list = None  # input posix paths
    filedata_list: list = None  # list of FileData objects
    slice_range: list = None

    def __post_init__(self):
        self.slice_range = [np.inf, -np.inf]
        if self.filepaths is not None:
            for f in self.filepaths:
                self.append(f)

    def __getitem__(self, index):
        #return getattr(self, name)
        return self.filedata_list[index]

    def __iter__(self):
        return iter(self.filedata_list)

    def __next__(self):
        return next(self.__iter__())

    def __lt__(self, other):
        return self.FrameOfRefUID < other.FrameOfRefUID

    def __gt__(self, other):
        return self.FrameOfRefUID > other.FrameOfRefUID

    def __eq__(self, other):
        return self.FrameOfRefUID == other.FrameOfRefUID

    def __len__(self):
        return len(self.filedata_list)

    def uid_check(self, dicomfile: DicomFile):
        if self.filedata_list is None:
            return True

        contained = self.filedata_list[0]
        patient = dicomfile.PatientID == contained.PatientID
        study = dicomfile.StudyUID == contained.StudyUID
        series = dicomfile.SeriesUID == contained.SeriesUID
        modality = dicomfile.Modality == contained.Modality
        ref_frame = dicomfile.FrameOfRefUID == contained.FrameOfRefUID

        return all([patient, study, series, modality, ref_frame])

    def append(self, filename: DicomFile):
        if filename.__class__ is not DicomFile:
            filename = DicomFile(filename)

        if True:  # self.uid_check(filename):
            if hasattr(filename, 'SliceLocation'):
                loc = filename.SliceLocation
                if loc < self.slice_range[0]:
                    self.slice_range[0] = loc
                if loc > self.slice_range[0]:
                    self.slice_range[1] = loc

            if self.filedata_list is None:
                self.filedata_list = [filename]
            else:
                self.filedata_list.append(filename)

    def pop(self):
        return self.filedata_list.pop()


# Make into an abstractclass because this is never instantiated
# Instead, this is generally inherited by all groups
class FileUtils:
    # A class to sort files and basic group management
    def __iter__(self):
        return iter(self.data.values())

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value):
        self.data[name] = value

    def __str__(self):
        name = self.__class__.__name__
        first_value = next(iter(self.data.values()))

        if first_value.__class__ is DicomGroup:
            output = name + '\n'
            item_iter = self.data.items()
            length = len(item_iter)
            for index, (key, value) in enumerate(item_iter):
                pad = '| ' + '  ' * self._depth
                if (index + 1) < length:
                    pad += '├─'
                else:
                    pad += '└─'
                output += pad + f'{key} : {len(value)} \n'
            return output

        self._tot_string += str(name) + ': ' + self._identifer + '\n'

        for index, (key, value) in enumerate(self.data.items()):
            pad = '| ' * max(0, self._depth - 1)
            if index < (len(self.data) - 1):
                pad += '├─'
            else:
                pad += '└─'
            self._tot_string += pad + value.__str__()

        return self._tot_string

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

    def digest_data(self):
        message = "Can only specify a path of list of files"
        assert (self.path is not None) != (self.all_files is not None), message

        if self.path is not None:
            self.all_files = glob(self.path + '/**/*.dcm', recursive=True)

        if self.filter_list is not None:
            self.all_files = self.file_filter()

        self.split()

        if self.name is None:
            f = self.all_files[0]
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            self.name = f[self._organize_by]

    def split(self):
        if self._files.__class__ is not DicomGroup:
            self._files = DicomGroup(self.all_files)

        fill_dict = {}
        files = deepcopy(self._files)

        while files:
            f = files.pop()
            key = str(f[self._organize_by])
            if key not in fill_dict:
                new_group = DicomGroup()
                new_group.append(f)
                fill_dict.update({key: new_group})
            else:
                fill_dict[key].append(f)

        fill_dict = dict(sorted(fill_dict.items()))

        for key, value in fill_dict.items():
            params = {'all_files': value,
                      '_identifer': key}
            if self.data is None:
                self.data = {key: value}
            if self._depth < 4:
                self.data[key] = self._child_type(**params)
            else:
                self.data[key] = value

    def file_filter(self):
        filtered_files = []
        files = deepcopy(self.all_files)
        while files:
            f = files.pop()
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            if f[self._organize_by] in self.filter_list:
                filtered_files.append(f)
        return filtered_files

    @property
    def group_id(self):
        if hasattr(self, '_identifer'):
            return self._identifer
        return None

    @property
    def name(self):
        return self.group_id
