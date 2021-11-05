import anytree
import os
import numpy as np
import pathlib
import pydicom
from . import utils
import multiprocessing
from anytree import RenderTree, NodeMixin
from anytree.iterators.levelordergroupiter import LevelOrderGroupIter
from copy import deepcopy, copy
from dataclasses import dataclass
from datetime import datetime
from pathos.pools import ProcessPool
from .processing import Reconstruction, Deconstruction, ImgAugmentations
from typing import Any, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor as ThreadPool


class GroupUtils(NodeMixin):
    """General utilities for all groups

    Args:
        NodeMixin (NodeMixin): AnyTree node mixin
        name (str, optional): name of the group. Defaults to None.
        files (list, optional): list of absolute paths
            to *.dcm files. Defaults to None.
        parent (Node, optional): The parent Node. Defaults to None.
        children (tuple, optional): A tuple of Nodes which are the
            children of the current Node. Defaults to None.
        include_series (bool, optional): Specifies if FrameOfRef points to
            Series (True) or Modality (False). Defaults to False.

    Methods:
        merge (NodeMixin): merges another group into current
        steal (NodeMixin): steals children from another group
        append (NodeMixin): appends children to group
        prune (str): prunes specified branch from tree
        flatten (None): flattens tree to one child per parent
    """
    def __init__(self, name=None, files=None, parent=None, children=None,
                 include_series=True, isodatetime=None, filter_by=None, writepath=None):
        super().__init__()
        self.name = name
        self.files = files
        self.parent = parent
        self.include_series = include_series
        self.isodatetime = isodatetime
        self.filter_by = filter_by
        self.writepath = writepath
        self._index = -1
        if self.files:
            self.files.sort()
        if children:
            self.children = children

    def __copy__(self) -> NodeMixin:
        """Creates copy of the current node, whereas deepcopy copies all desendants

        Returns:
            NodeMixin: A copy of the current node
        """
        new = type(self)('')
        for key, value in self.__dict__.items():
            if key == '_NodeMixin__children':
                new.__dict__.update({key: []})
            else:
                new.__dict__.update({key: value})
        return new

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def __next__(self):
        if self._index == (len(self.children) - 1):
            raise StopIteration
        else:
            self._index += 1
            return self.children[self._index]

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        output = []
        for pre, fill, node in RenderTree(self, childiter=self._str_sort):
            line = f'{pre}{node.__class__.__name__}:'
            line += f'\033[1m\033[91m{node.name}\033[0m'
            if node.__class__.__name__ == 'Modality':
                line += '\033[1m\033[91m' + node.__str__() + '\033[0m'
            output.append(line)
        return '\n'.join(output)

    def _str_sort(self, items: list) -> list:
        return sorted(items, key=lambda item: item.name)

    def _digest(self):
        """Digests the file paths, building the tree
        """
        while self.files:
            f = self.files.pop()
            if f.__class__ is not DicomFile:
                f = DicomFile(f)
            self._add_file(f)

    def _digestfn(self, dicomfile: 'DicomFile') -> NodeMixin:
        """Function to digest a dicom file, sorting it into the tree

        Args:
            dicomfile (DicomFile): Dicom file to add to the tree

        Returns:
            NodeMixin: Child type of the root node
        """
        if dicomfile.__class__ is not DicomFile:
            dicomfile = DicomFile(dicomfile)
        if type(dicomfile) is pydicom.dataset.Dataset:
            dicomfile = DicomFile(dicomfile.SOPInstanceUID, dcm_obj=dicomfile)

        if self._filter_check(dicomfile):
            key = str(dicomfile[self._organize_by])
            found = False
            for child in self.children:
                if key == child.name:
                    child._add_file(dicomfile)
                    found = True

            if not found:
                dt = dicomfile.DateTime.isoformat(self._child_type)
                child_params = {'name': key,
                                'files': [dicomfile],
                                'parent': self,
                                'include_series': self.include_series,
                                'filter_by': self.filter_by,
                                'isodatetime': dt}
                return self._child_type(**child_params)

    def _find_leaf(self, combined: NodeMixin, modality: 'Modality') -> NodeMixin:
        """When merging into a unified tree, the proper leaf must be identified

        Args:
            combined (NodeMixin): The combined or unified tree
            modality (Modality): A modality to add or merge into the tree

        Returns:
            NodeMixin: The combined or unifed tree
        """
        current = combined
        previous = None
        for ancs in modality.ancestors:
            # Find the deepest common ancestor
            previous = current
            try:
                current = anytree.search.findall(current, filter_=lambda x: x.name == ancs.name)[-1]
            except Exception:
                current = anytree.search.find(current, filter_=lambda x: x.name == ancs.name)

            if current is None:  # not in tree
                previous.adopt(ancs)
                break
            elif current._organize_by == 'Modality':  # at bottom, find series or make new
                found = False
                for series in current:
                    if series.name == modality.name and series is not modality:
                        found = True
                        for f in modality.dicoms:
                            series._add_file(f)
                if not found:
                    modality.parent = current
        return combined

    def _clean_up_children(self, children: list) -> NodeMixin:
        """Following multithreaded sorting, trees need to be merged into one

        Args:
            children (list): A list of trees with reconstructed volumes or files

        Returns:
            NodeMixin: A unified tree
        """
        combined = self.__class__(self.name)
        for child in children:
            within = [child.name == x.name for x in combined]
            if not any(within) or len(within) == 0:
                combined.adopt(child)
            else:
                for modality in child.iter_modalities():
                    combined = self._find_leaf(combined, modality)
        return combined

    def _multithread_digest(self) -> None:
        """Multithreaded tree digestion for greater IOPs and faster tree construction
        """
        with ProcessPool() as P:
            children = [x for x in list(P.map(self._digestfn, self.files)) if x]
        ProcessPool().clear()

        # Need to iterate through and merge any duplicates
        cleaned = self._clean_up_children(children)

        for child in cleaned:
            self.adopt(child)
        self.files = None

    def _filter_check(self, dicomfile: object) -> bool:
        """Function to identify if the node is unique or existing

        Args:
            dicomfile (object): Dicom file to check uniqueness

        Returns:
            bool: Returns if the file is unique, needing a new node, or not
        """
        def date_check(date, dtype):
            condition0 = str(date) in self.filter_by[dtype]
            condition1 = int(date) in self.filter_by[dtype]
            return not (condition0 or condition1)

        # We want to return true if its good to add
        if self.filter_by:
            conditions = {'mrn': True,
                          'study': True,
                          'series': True,
                          'modality': True}
            if 'PatientID' in self.filter_by:
                mrn = dicomfile.PatientID
                conditions['mrn'] = (mrn in self.filter_by['PatientID'])
            if 'StudyDate' in self.filter_by:
                date = dicomfile.DateTime.StudyDate
                conditions['study'] = date_check(date, 'StudyDate')
            if 'SeriesDate' in self.filter_by:
                date = dicomfile.DateTime.SeriesDate
                conditions['series'] = date_check(date, 'SeriesDate')
            if 'Modality' in self.filter_by:
                modality = dicomfile.Modality
                conditions['modality'] = modality in self.filter_by['Modality']
            return all(conditions.values())
        return True

    def _add_file(self, dicomfile: object, volumes: bool = False) -> None:
        """Adds a file to the file tree

        Args:
            dicomfile (DicomFile): A DicomFile object
        """
        if type(dicomfile) is pydicom.dataset.Dataset:
            dicomfile = DicomFile(dicomfile.SOPInstanceUID, dcm_obj=dicomfile)

        if self._filter_check(dicomfile):
            key = str(dicomfile[self._organize_by])
            found = False
            for child in self.children:
                if key == child.name:
                    child._add_file(dicomfile)
                    found = True

            if not found:
                dt = dicomfile.DateTime.isoformat(self._child_type)
                child_params = {'name': key,
                                'files': [dicomfile],
                                'parent': self,
                                'filter_by': self.filter_by,
                                'include_series': self.include_series,
                                'isodatetime': dt}
                self._child_type(**child_params)

    @property
    def dirname(self) -> None:
        if hasattr(self, '_dirname'):
            return self._dirname
        return repr(self) + '_' + self.name

    @dirname.setter
    def dirname(self, name: str) -> None:
        self._dirname = name

    @property
    def datename(self) -> None:
        if hasattr(self, '_datename'):
            return self._datename
        if not self.isodatetime:
            return '_'.join([self.name, repr(self), 'NoDateTime_'])
        return '_'.join([repr(self), self.name, self.isodatetime])

    @datename.setter
    def datename(self, name: str) -> None:
        self._datename = name

    def merge(self, other: NodeMixin) -> None:
        """Merges two groups

        Args:
            other (NodeMixin): Merged into primary group
        """
        other.parent = self.parent

    def steal(self, other: NodeMixin) -> None:
        """Steals children from one parent

        Args:
            other (NodeMixin): Parent who loses children

        Notes:
            Porter et al. do not condone kidnapping
        """
        self.children += other.children

    def append(self, others: NodeMixin) -> None:
        """Appends children onto parent

        Args:
            others (NodeMixin): Children to be appended
        """
        for other in others:
            other.parent = self

    def prune(self, childname: str) -> None:
        """Prunes branch from tree

        Args:
            childname (str): Name of branch to prune
        """
        for child in self.children:
            if child.name in childname:
                child.parent = None

    def adopt(self, child: NodeMixin) -> None:
        """Adopts a child from one tree to the current tree

        Args:
            child (NodeMixin): Node to adopt to tree
        """
        child.parent = self

    def abandon(self) -> None:
        """Parent abandons all children, alternative to prune
        """
        self._NodeMixin__children = []

    def integrate(self, child: NodeMixin) -> None:
        """Integrates a child into a tree. If the node ancestors do not exist, they are created

        Args:
            child (NodeMixin): Node to integrate into tree
        """
        if self._child_type is type(child):  # Child is directly adoptable
            child.parent = self
        else:
            current = self
            previous = None
            for a in child.ancestors[1:]:
                # find matching ancestor
                previous = current
                if current is None:
                    break

                try:
                    current = anytree.search.findall(current, filter_=lambda x: x.name == a.name)[-1]
                except Exception:
                    current = anytree.search.find(current, filter_=lambda x: x.name == a.name)

                if current is None:  # ancestor not in tree
                    new = copy(a)
                    new._NodeMixin__parent = None
                    new.parent = previous
                    new.integrate(child)
                elif current._child_type is type(child):  # adopt to ancestor
                    child.parent = current
                    break  # Stop iteration now that its adopted

    def flatten(self) -> None:
        """Flatten results in each parent having one child, except for
            the Modality group. Flattened with specified group as root
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

    def pop(self) -> NodeMixin:
        """Pops a child from the parent

        Returns:
            NodeMixin: Returns the first child

        Notes:
            The item is not deleted, instead its parent is
                changed to None, severing the relationship
        """
        child = self.children[0]
        child.parent = None
        return child

    def save_tree(self, path: str, prefix: str = 'group') -> None:
        """[saves a copy of the tree to a specified location, ordered
            the same as the tree layout]

        Args:
            path (str): [absolute path to write to file tree]
            prefix (str, optional): [Specifies directory prefix as 'group',
                'date' or None]. Defaults to 'group'.
        """
        utils.save_tree(self, path, prefix)

    def save_dicoms(self, path: str, prefix: str = 'group') -> None:
        """Saves a copy of the dicom tree to a specified location,
            ordered the same as the tree layout

        Args:
            path (str): Absolute path to write to the file tree
            prefix (str, optional): Specifies directory prefix as 'group'.
                'date' or None. Defaults to 'group'.
        """
        dicoms = self.split_dicoms()
        dicoms.save_tree(path, prefix)

    def save_volumes(self, path: str, prefix: str = 'group') -> None:
        """Saves a copy of the volume tree to a specified location,
            ordered the same as the tree layout

        Args:
            path (str): Absolute path to write to the file tree
            prefix (str, optional): Specifies directory prefix as 'group'.
                'date' or None. Defaults to 'group'.
        """
        volumes = self.split_volumes()
        volumes.save_tree(path, prefix)

    def only_dicoms(self) -> bool:
        """True if tree only contains dicom leaves

        Returns:
            bool: True if tree only contains dicom leaves
        """
        return not self.has_volumes()

    def only_volumes(self) -> bool:
        """True if tree only contains volume leaves

        Returns:
            bool: True if tree only contains volume leaves
        """
        return not self.has_dicoms()

    def has_dicoms(self):
        """Determines if tree contains dicoms

        Returns:
            bool: True if tree contains dicoms
        """
        return bool(next(self.iter_dicoms(), False))

    def has_volumes(self):
        """Determines if tree contains volumes

        Returns:
            bool: True if tree contains volumes
        """
        return bool(next(self.iter_volume_data(), False))  # iter_volume_data

    def iter_patients(self) -> object:
        def filterpatient(node):
            return type(node) is Patient
        iterer = LevelOrderGroupIter(self, filter_=filterpatient)
        patients = [t for t in iterer if t]
        return iter(*patients)

    def iter_studies(self) -> object:
        def filterstudy(node):
            return type(node) is Study
        iterer = LevelOrderGroupIter(self, filter_=filterstudy)
        studies = [t for t in iterer if t]
        return iter(*studies)

    def iter_frames(self, return_count: bool = False) -> object:
        """Iterates through each FrameOfRef

        Args:
            return_count (bool, optional): Returns number of frames. Defaults to False.

        Returns:
            object: FrameOfRef object iterator
        """
        def filterframe(node):
            return type(node) is FrameOfRef
        iterer = LevelOrderGroupIter(self, filter_=filterframe)
        frames = [t for t in iterer if t]
        if return_count:
            return (len(*frames), iter(*frames))
        return iter(*frames)

    def iter_series(self) -> object:
        def filterseries(node):
            return type(node) is Series
        iterer = LevelOrderGroupIter(self, filter_=filterseries)
        series = [t for t in iterer if t]
        return iter(*series)

    def iter_modalities(self) -> object:
        """Iterates through each Modality

        Returns:
            object: Modality object iterator
        """
        def filtermod(node):
            return type(node) is Modality
        iterer = LevelOrderGroupIter(self, filter_=filtermod)
        mods = [t for t in iterer if t]
        if not mods == [None] * len(mods):
            return iter(*mods)

    def iter_dicoms(self) -> dict:
        """Iterates over dicom groups

        Returns:
            dict: dicom data dict from each modality

        Yields:
            Iterator[dict]: iterator for each modality
        """
        for mod in self.iter_modalities():
            if mod.dicoms_data:
                yield mod.dicoms_data

    def iter_volume_data(self) -> dict:
        """[Iterates over the reconstructed volumes]

        Returns:
            dict: [dict of volume data for each modality]

        Yields:
            Iterator[dict]: [returns dict of ReconstructedVolume or ReconstructedFile]
        """
        for mod in self.iter_modalities():
            if mod.volumes_data:
                yield mod.volumes_data

    def iter_volumes(self) -> Union['ReconstructedVolume', 'ReconstructedFile']:
        """Iterates over each individual volume in a cohort

        Yields:
            Union[ReconstructedVolume, ReconstructedFile]: Returns each volume file
        """
        for vol_data in self.iter_volume_data():
            for vols in vol_data.values():
                for vol in vols:
                    yield vol

    def iter_volume_frames(self) -> list:
        """Iterates through the frame of references

        Returns:
            list: list of all volumes in a FrameOfRef

        Yields:
            Iterator[list]: returns a list of all volumes

        Notes:
            TODO: Decide if a different data structure is better
        """
        for frame in self.iter_frames():
            vols = {}
            for vol in frame.iter_volume_data():
                vols.update(vol)
            yield vols

    def iter_image_volume_files(self, modality: str):
        """Iterates through image volume files

        Args:
            modality (str): Modality type to iterate through

        Yields:
            Union[ReconstructedVolume, ReconstructedFile]: Reconstructed volume or
                file of specified image
        """
        for mod in self.iter_modalities():
            if mod.name == modality:
                for volfiles in mod.volumes_data.values():
                    for volfile in volfiles:
                        yield volfile

    def iter_struct_volume_files(self):
        """[Iterates through RTSTRUCT volume files]

        Yields:
            Union[ReconstructedVolume, ReconstructedFile]: [Reconstructed volume or
                file of RTSTRUCT]
        """
        for mod in self.iter_modalities():
            if mod.name == 'RTSTRUCT':
                for volfiles in mod.volumes_data.values():
                    for volfile in volfiles:
                        yield volfile

    def clear_dicoms(self) -> None:
        """[Clears the dicoms from the tree]
        """
        for mod in self.iter_modalities():
            mod.clear_dicoms()

    def clear_volumes(self) -> None:
        """[Clears the volumes from the tree]
        """
        for mod in self.iter_modalities():
            mod.clear_volumes()

    def split_trees(self) -> tuple:
        """[Split the dicom and volume trees]

        Returns:
            tuple: [(dicom_tree, volume_tree)]

        Notes:
            Inefficient to copy twice, fix in the future
        """
        tree1 = deepcopy(self)
        tree1.clear_volumes()
        tree2 = deepcopy(self)
        tree2.clear_dicoms()
        return (tree1, tree2)

    def split_dicoms(self) -> object:
        """Split the dicom tree from the volume tree

        Returns:
            object: Returns a tree with only dicoms at the leaves

        Notes:
            Inefficient to copy twice, fix in the future
        """
        dicomtree = deepcopy(self)
        dicomtree.clear_volumes()
        self.clear_dicoms()
        return dicomtree

    def split_volumes(self) -> object:
        """Split the volume tree from the dicom tree

        Returns:
            object: Returns a tree with only volumes at the leaves

        Notes:
            Inefficient to copy twice, fix in the future
        """
        voltree = deepcopy(self)
        self.clear_volumes()
        voltree.clear_dicoms()
        return voltree

    def volumes_to_files(self) -> None:
        """Converts all volumes to files
        """
        for vol in self.iter_volumes():
            if type(vol) is ReconstructedVolume:
                vol.convert_to_file()
            else:
                utils.colorwarn(f'{vol.name} already {type(vol)}')

    def files_to_volumes(self) -> None:
        """Converts all files to volumes

        Notes:
            This may require a large amount of memory
        """
        for vol in self.iter_volumes():
            if type(vol) is ReconstructedFile:
                vol.load_array()
            else:
                utils.colorwarn(f'{vol.name} already {type(vol)}')

    def _recon_to_disk(self, return_mods: bool = False, path: str = None) -> None:
        """Reconstructs self, writes volumes to disk

        Args:
            return_mods (bool, optional): [Returns the modality pairs]. Defaults to False.
        """
        def recon_fn(frame):
            return frame.recon(in_memory=False, path=path)

        iterator = self.iter_frames()
        with ProcessPool(nodes=4) as P:
            frame_group = list(P.map(recon_fn, iterator))
            P.close()
            P.join()
            P.clear()

        mods_ptrs = utils.flatten_list(frame_group)
        if return_mods:
            return mods_ptrs
        else:
            utils.insert_into_tree(self, mods_ptrs)

    def _recon_to_memory(self):
        """Reconstructs volumes and stores in memory

        Notes:
            Not recommended for large data sets (N>10), only useful for
                small sample inference applications or write limited hardware
        """
        total_frames, iterator = self.iter_frames(return_count=True)
        if total_frames > 10:
            source = 'DICOManager/groupings.py'
            message = '.recon(in_memory=False, parallelize=True) recommended for large datasets'
            utils.colorwarn(message, source)
        for frame in iterator:
            frame.recon(in_memory=True)

    def recon(self, in_memory: bool = False, parallelize: bool = True,
              path: str = None, filter_by: dict = None, *args, **kwargs) -> None:
        """Reconstruction of a patient or cohort

        Args:
            in_memory (bool, optional): Saves the volumes into memory if True
                and onto disk if False. Defaults to False.
            parallelize (bool, optional): Parallelize onto additional CPU cores,
                runs faster this way. Defaults to True.

        Notes:
            in_memory=True is only recommended for small datasets or uses where the
                hardware is write-limited. Loading large amounts of data into memory
                may cause the system to swap and become exceedingly slow.
        """
        if path:
            self.writepath = path

        if filter_by is not None:
            if type(filter_by) is list:
                filter_by = {'StructName': filter_by}
            elif 'StructName' not in filter_by:
                raise TypeError('Either a list, or dict with key \'StructName\' must be given')

            if type(self) is FrameOfRef:
                self.add_name_filter(names=filter_by['StructName'])
            else:
                for frame in self.iter_frames():
                    frame.add_name_filter(names=filter_by['StructName'])

        if in_memory and not parallelize:
            self._recon_to_memory()
        elif parallelize:
            self = utils.threaded_recon(self, path=path)
            if in_memory:
                self.files_to_volumes()
        else:
            return self._recon_to_disk(path=path)

    def apply_tool(self, toolset: list, path: str = None,
                   mod: 'Modality' = None, warn: bool = True) -> None:
        """A single threaded application of a list of tool functions

        Args:
            toolset (list): List of tool functions
            path (str, optional): Path to write volumes. Defaults to None.
            mod (Modality, optional): Modality to apply function to. Defaults to self.
            warn (bool, optional): Raise warning when applied. Defaults to True.

        Warnings:
            If path is not specified, the previously reconstructed volumes will be
                overwritten by the newly modified volumes
        """
        if warn:
            message = 'Applied tools will overwrite volumes unless path is specified'
            utils.colorwarn(message)
        if type(toolset) is not list:
            toolset = list(toolset)
        if not path:
            path = self.writepath

        if mod is None:
            it = self.iter_volume_data()
        else:
            it = mod.iter_volume_data()

        for volume in it:
            for name, reconfiles in volume.items():
                newfiles = []
                for f in reconfiles:
                    for tool in toolset:
                        f = tool(f, path)
                    newfiles.append(f)
                volume[name] = newfiles

    def apply_tools(self, toolset: list, path: str = None, nthreads: int = None) -> None:
        """Multithreaded application of a list of tool functions applied

        Args:
            toolset (list): List of tool functions
            path (str, optional): Path to write volumes. Defaults to previous save path.
            nthreads (int, optional): Number of concurrent threads. Defaults to nCPUs // 2.

        Returns:
            None: Changes to the cohort are completed in place]

        Warnings:
            If path is not specified, the previously reconstructed volumes will be
                overwritten by the newly modified volumes

        Notes:
            nthreads = cpus / 16 was chosen to limit memory footprint. Increasing this
                number may not translate to a decreased runtime if disk I/O is saturated
        """
        if type(toolset) is not list:
            toolset = list(toolset)
        if not path:
            path = self.writepath
        if not nthreads:
            nthreads = multiprocessing.cpu_count() // 2

        # Multithreaded
        def toolfn(toolset, path):
            def wrapper(mod):
                self.apply_tool(toolset=toolset, path=path, mod=mod, warn=False)
                return None
            return wrapper

        message = 'Applied tools will overwrite volumes unless path is specified'
        utils.colorwarn(message)

        fn = toolfn(toolset, path)
        it = self.iter_modalities()
        with ThreadPool(max_workers=nthreads) as P:
            _ = list(P.map(fn, it))
            ThreadPool().shutdown()

    def _incomplete(self, group: str, exact: bool = False, contains: dict = None) -> bool:
        """Determines if group is incomplete as defined by contains

        Args:
            group (str): Group to determine completeness
            exact (bool, optional): Group must contain exactly what is specified in contains,
                otherwise the group can contain more than contains and be valid. Defaults to False.
            contains (dict, optional): A dictionary with keys of the group type and the
                corresponding features. For example:
                        contains = {'Modality': ['CT', 'MR', 'MR', 'RTSTRUCT']}
                would mean with exact=True, the group must contain exactly 1 CT, 2 MR and
                1 RTSTRUCT. With exact=False, the group could contain any number of CT, MR,
                RTSTRUCT as long as at least one of each file type is present. Defaults to None.

        Returns:
            bool: Returns if the group is complete]
        """
        if contains is None:
            contains = self.fitler_by
        decendant_types = utils.decendant_types(group)
        passes_dtype = []  # Checks each decendant level to confirm passes
        for dtype in decendant_types:
            if dtype not in contains:
                passes_dtype.append(True)
                continue
            needed = copy(contains[dtype])
            fields = []
            for modality in self.iter_modalities():
                f = modality.dicoms[0]
                if dtype == 'Patient':
                    fields.append(f.PatientID)
                elif dtype == 'FrameOfRef':
                    fields.append(f.FrameOfRefUID)
                elif dtype == 'Study':
                    fields.append(f.StudyUID)
                elif dtype == 'Series':
                    fields.append(f.SeriesUID)
                else:
                    fields.append(modality.name)
            if exact:
                if len(fields) != len(needed):
                    passes = False
                else:
                    passes = sorted(fields) == sorted(needed)
            else:
                setneeded = set(needed)
                setfields = set(fields)
                passes = setneeded.issubset(setfields)
            passes_dtype.append(passes)
        return not all(passes_dtype)

    def pull_incompletes(self, group: str = 'Patient', exact: bool = False,
                         contains: dict = None, cleaned: bool = False) -> NodeMixin:
        """Determines if group is incomplete as defined by contains, returns the excluded
            groups from the tree. Original tree is modified in place

        Args:
            group (str): Group to determine completeness, valid groups are Patient, FrameOfRef,
                Study, Series. Defaults to Patient.
            exact (bool, optional): Group must contain exactly what is specified in contains,
                otherwise the group can contain more than contains and be valid. Defaults to False.
            contains (dict, optional): A dictionary with keys of the group type and the
                corresponding features. For example:
                        contains = {'Modality': ['CT', 'MR', 'MR', 'RTSTRUCT']}
                would mean with exact=True, the group must contain exactly 1 CT, 2 MR and
                1 RTSTRUCT. With exact=False, the group could contain any number of CT, MR,
                RTSTRUCT as long as at least one of each file type is present. If contains is
                not specified, the filter_by parameter of the parent group will be used, as
                defined at group creation. If that parameter is None, an AttributeError
                will be raised. Defaults to None.
            cleaned (bool, optional): Applies remove_empty_branches() following splitting

        Returns:
            NodeMixin: Tree of the excluded groups from the original tree]

        Raises:
            AttributeError: Raised if both self.filter_by and contains are None]
        """
        # fliter_by will default to the original grouping, if specified, otherwise will warn
        if self.filter_by is None and contains is None:
            raise AttributeError('No <tree>.filter_by or specified filter_by')

        incomplete_tree = copy(self)

        if group == 'Patient':
            for patient in self.iter_patients():
                if patient._incomplete(group, exact, contains):
                    incomplete_tree.integrate(patient)
        elif group == 'FrameOfRef':
            for frame in self.iter_frames():
                if frame._incomplete(group, exact, contains):
                    incomplete_tree.integrate(frame)
        elif group == 'Study':
            for study in self.iter_studies():
                if study._incomplete(group, exact, contains):
                    incomplete_tree.integrate(study)
        elif group == 'Series':
            for series in self.iter_series():
                if series._incomplete(group, exact, contains):
                    incomplete_tree.integrate(series)
        else:
            types = 'Patient, FrameOfRef, Study or Series'
            message = f'Group level must be specified as {types} to check for completeness'
            raise AttributeError(message)

        if cleaned:
            self.remove_empty_branches()

        return incomplete_tree

    def remove_empty_branches(self, prune_modalities: bool = False) -> None:
        """Removes any empty branches from a tree. An empty branch is defined
            as a branch with no children. By default, modalities are excluded.

        Args:
            prune_modalities (bool, optional): If specified as True, modalities
                with no dicoms and no volumes will be pruned as well. Defaults to False.
        """
        for child in self.children:
            if type(child) is Modality:
                if prune_modalities:
                    if not child.dicoms_data and not child.volumes_data:
                        self.prune(child.name)
                else:
                    break
            elif len(child.children) == 0:
                self.prune(child.name)
            else:
                child.remove_empty_branches()


class ReconstructedVolume(GroupUtils):
    # These properties might be unnecessary
    ct = property(utils.mod_getter('CT'), utils.mod_setter('CT'))
    nm = property(utils.mod_getter('NM'), utils.mod_setter('NM'))
    mr = property(utils.mod_getter('MR'), utils.mod_setter('MR'))
    pet = property(utils.mod_getter('PET'), utils.mod_setter('PET'))
    dose = property(utils.mod_getter('RTDOSE'), utils.mod_setter('RTDOSE'))
    struct = property(utils.mod_getter('RTSTRUCT'), utils.mod_setter('RTSTRUCT'))

    def __init__(self, dcm_header: pydicom.dataset.Dataset,
                 dims: utils.VolumeDimensions, parent: object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dcm_header = dcm_header
        self.name = dcm_header.SeriesInstanceUID
        self.dims = deepcopy(dims)
        self.volumes = {}
        self.ImgAugmentations = ImgAugmentations()
        self._parent = deepcopy(parent)  # We don't want to actually add it to the tree
        self.remove_parent_data()
        self._digest()

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        setattr(self, name, value)

    def __str__(self):
        middle = []
        for index, key in enumerate(self.volumes):
            middle.append(f' {len(self.volumes[key])} files')
        output = ' [' + ','.join(middle) + ' ]'
        return output

    def _rename(self, name: str):
        """DICOM RTSTRUCTs can have non-unique names, so we need to rename
            these functions to be dictionary compatible

        Args:
            name (str): Name of the found RTSTRUCT

        Returns:
            str]: Name + # where number is the unique occupance
        """
        i = 0
        while True:
            temp = name + i
            if temp not in self.data:
                break
            i += 1
        return temp

    def _digest(self):
        """Pulls relevant information into the class]
        """
        ds = self.dcm_header
        self.PatientID = ds.PatientID
        self.StudyInstanceUID = ds.StudyInstanceUID
        self.SeriesInstanceUID = ds.SeriesInstanceUID
        self.Modality = ds.Modality
        self.SOPInstanceUID = ds.SOPInstanceUID

        if type(ds) is pydicom.dataset.FileDataset:
            self.DateTime = DicomDateTime(ds)

        if hasattr(ds, 'FrameOfReferenceUID'):
            self.FrameOfRefUID = ds.FrameOfReferenceUID
        elif hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
            ref_seq = ds.ReferencedFrameOfReferenceSequence
            self.FrameOfRefUID = ref_seq[0].FrameOfReferenceUID
        elif hasattr(ds, 'FrameOfRefUID'):
            self.FrameOfRefUID = ds.FrameOfRefUID
        else:
            self.FrameOfRefUID = None
        self.dcm_header = None

    def _pull_header(self) -> dict:
        """Pulls header information from the data structure to export

        Returns:
            dict: dictionary of associated header information
        """
        fields = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
                  'Modality', 'SOPInstanceUID', 'FrameOfRefUID']
        header = {}
        for field in fields:
            header.update({field: self[field]})
        return header

    def _generate_filepath(self, prefix: str = 'group') -> str:
        """Generates a filepath, in tree hierarchy to save the file

        Args:
            prefix (str, optional): Directory naming convention, similar to
                save_tree(). Defaults to 'group'.

        Returns:
            str: the generated filepath
        """
        parents = [self._parent]
        temp = self._parent
        while hasattr(temp.parent, 'parent'):
            parents.append(temp.parent)
            temp = temp.parent

        if prefix == 'group':
            names = [p.dirname for p in parents]
        elif prefix == 'date':
            names = [p.dirname for p in parents]
        else:
            names = [p.name for p in parents]

        names.reverse()
        names[0] += '_volumes'
        return '/'.join(names) + '/'

    def is_file(self):
        return False

    def remove_parent_data(self) -> None:
        if self._parent.has_dicoms():
            self._parent.clear_dicoms()
        if self._parent.has_volumes():
            self._parent.clear_volumes()

    def add_vol(self, name: str, volume: np.ndarray):
        """Add a volume array to the data structure

        Args:
            name (str): Name of the volume array
            volume (np.ndarray): Array
        """
        if name in self.volumes:
            name = self._rename(name)

        self.volumes.update({name: volume})

    def add_structs(self, structdict: dict):
        """Add a RTSTRUCT array to the data structure

        Args:
            structdict (dict): A dictionary of structure names and
                their corresponding volume arrays
        """
        for name, volume in structdict.items():
            self.add_vol(name, volume)

    @property
    def shape(self):
        return self.dims.shape

    @property
    def is_struct(self):
        return self.Modality == 'RTSTRUCT'

    def load_array(self):
        # Included to prevent unnecessary errors
        pass

    def export(self, include_augmentations: bool = True, include_dims: bool = True,
               include_header: bool = True, include_datetime: bool = True) -> dict:
        """Export the ReconstructedVolume array

        Args:
            include_augmentations (bool, optional): Include augmentations. Defaults to True.
            include_dims (bool, optional): Include dict of volume dimensions. Defaults to True.
            include_header (bool, optional): Include dict of UID values. Defaults to True.
            include_datetime (bool, optional): Include dict of dates and times. Defaults to True.

        Returns:
            dict: A dictionary of volumes and specified additional information
        """
        export_dict = {}
        export_dict.update({'name': self.name})
        export_dict.update({'volumes': dict(sorted(self.volumes.items()))})
        if include_augmentations:
            export_dict.update({'augmentations': self.ImgAugmentations.as_dict()})
        if include_dims:
            export_dict.update({'dims': self.dims.as_dict()})
        if include_header:
            export_dict.update({'header': self._pull_header()})
        if include_datetime:
            export_dict.update({'DateTime': self.DateTime.export()})
        return export_dict

    def convert_to_file(self, path: str = None, save: bool = True) -> None:
        """Converts ReconstructedVolume to ReconstructedFile and saves
            the volume array to ~/tree/format/SeriesInstanceUID.npy
        """
        populate = {'name': self.name,
                    'dims': self.dims,
                    'header': self._pull_header(),
                    'augmentations': self.ImgAugmentations.export(),
                    'DateTime': self.DateTime.export()}
        filepath = self.save_file(save_dir=path, return_loc=True, save=save)
        self.volumes = None
        self.__class__ = ReconstructedFile
        self.__init__(filepath, self._parent, populate)
        self.remove_parent_data()

    def save_file(self, save_dir: str = None, filepath: str = None,
                  return_loc: bool = False, save: bool = True) -> Union[None, pathlib.PosixPath]:
        """Save the reconstructed volume and associated information to a .npy pickled file

        Args:
            save_dir (str, optional): Directory to save file to, defaults to ~/. Defaults to None.
            filepath (str, optional): Filepath to save file tree. Defaults to None.
            return_loc (bool, optional): Returns saved filepath location. Defaults to False.
            save (bool, optional): Determines if the file is actually saved or just converted

        Raises:
            TypeError: Raised if save_dir and filepath are specified

        Returns:
            [None, pathlib.Path]: None or pathlib.Path to saved file location

        Notes:
            In some cases, we simply want to load the array, read it and not rewrite it to disk,
                in those instances, we should use save=False to reduce needless disk I/O
        """
        if save_dir and filepath:
            raise TypeError('Either a save directory or a full filepath')
        if not filepath:
            filepath = self._generate_filepath()
        if save_dir:
            fullpath = pathlib.Path(save_dir) / filepath
            self._prior_save_dir = save_dir
        elif self._prior_save_dir:
            fullpath = pathlib.Path(self._prior_save_dir) / filepath
        else:
            fullpath = pathlib.Path.home() / filepath

        pathlib.Path(fullpath).mkdir(parents=True, exist_ok=True)
        output = copy(self.export())

        if save:
            np.save(fullpath / self.SeriesInstanceUID, output)

        if return_loc:
            return fullpath / (self.SeriesInstanceUID + '.npy')


class ReconstructedFile(GroupUtils):
    """A file to the saved file, as opposed to the file in memory (ReconstructedVolume).

        When reconstructing and saving to disk, we will generate the file, write to disk and
        add this leaf type within the tree. If we want to load that into memory, we will simply
        replace this node type with the type of ReconstructedVolume
    """
    def __init__(self, filepath: str, parent: object, populate: dict = None, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.filepath = filepath
        self.populate = populate
        self._parent = parent
        self.remove_parent_data()
        self._digest()

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        setattr(self, name, value)

    def __str__(self):
        return ' [ 1 file path to volume ]'

    def _digest(self):
        """Generates a ReconstructedFile from a ReconstructedVolume, either from a dict or loaded
        """
        if self.populate:
            self.name = self.populate['name']
            self.dims = self.populate['dims']
            self.ImgAugmentations = ImgAugmentations().from_dict(self.populate['augmentations'])
            self.header = self.populate['header']
            self.DateTime = DicomDateTime().from_dict(self.populate['DateTime'])
        else:
            ds = np.load(self.filepath, allow_pickle=True, mmap_mode='r').item()
            self.name = ds['name']
            self.dims = ds['dims']
            augs = ImgAugmentations().from_dict(ds['augmentations'])
            self.ImgAugmentations = augs
            self.header = ds['header']
            self.DateTime = DicomDateTime().from_dict(ds['DateTime'])
        for name, value in self.populate['header'].items():
            self[name] = value

    def is_file(self):
        return True

    def convert_to_file(self):
        # Included to prevent unnecessary errors
        pass

    def remove_parent_data(self) -> None:
        if self._parent.has_dicoms():
            self._parent.clear_dicoms()
        if self._parent.has_volumes():
            self._parent.clear_volumes()

    def load_array(self):
        """Loads volume array from disk into memory, converts type to ReconstructedVolume
        """
        ds = np.load(self.filepath, allow_pickle=True).item()
        ds_header = utils.dict_to_dataclass(ds['header'])
        self.__class__ = ReconstructedVolume
        self.__init__(ds_header, self.dims, self._parent)
        self.volumes = ds['volumes']
        self.DateTime = DicomDateTime().from_dict(ds['DateTime'])
        self.ImgAugmentations = ImgAugmentations().from_dict(ds['augmentations'])
        self.remove_parent_data()


class DicomFile(GroupUtils):
    """Group level for individual Dicom files, pulls releveant header data out

    Args:
        filepath (str): absolute file path to the *.dcm file
    """
    SelfType = TypeVar('DicomFile')

    def __init__(self, filepath: str,
                 dcm_obj: pydicom.dataset.Dataset = None, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.name = os.path.basename(filepath)
        self.filepath = filepath
        self.dcm_obj = dcm_obj
        self._digest()

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __lt__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(other, 'SliceLocation'):
            return self.SliceLocation < other.SliceLocation
        return self.InstanceUID < other.InstanceUID

    def __gt__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(other, 'SliceLocation'):
            return self.SliceLocation > other.SliceLocation
        return self.FrameOfRefUID > other.FrameOfRefUID

    def __eq__(self, other: SelfType):
        if hasattr(self, 'SliceLocation') and hasattr(other, 'SliceLocation'):
            return self.SliceLocation == other.SliceLocation
        return self.FrameOfRefUID == other.FrameOfRefUID

    def _pull_info(self, ds: pydicom.dataset.FileDataset) -> None:
        """Pulls the info from the pydicom dataset

        Args:
            ds (pydicom.dataset.FileDataset): [DICOM to remove relevant data from]
        """
        self.PatientID = ds.PatientID
        self.StudyUID = ds.StudyInstanceUID
        self.SeriesUID = ds.SeriesInstanceUID
        self.Modality = ds.Modality
        self.InstanceUID = ds.SOPInstanceUID
        self.DateTime = DicomDateTime(ds)
        # For reconstruction

        if hasattr(ds, 'FrameOfReferenceUID'):
            self.FrameOfRefUID = ds.FrameOfReferenceUID
        elif hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
            ref_seq = ds.ReferencedFrameOfReferenceSequence
            self.FrameOfRefUID = ref_seq[0].FrameOfReferenceUID
        else:
            self.FrameOfRefUID = None

        if hasattr(ds, 'Columns'):
            # Need to standardize the nomenclature
            self.cols = ds.Columns
            self.rows = ds.Rows

        if hasattr(ds, 'SliceLocation'):
            self.SliceLocation = float(ds.SliceLocation)
        elif hasattr(ds, 'ImagePositionPatient'):
            self.SliceLocation = float(ds.ImagePositionPatient[-1])
        self.dcm_obj = None

    def _digest(self):
        """Digestion of the pydicom.dataset.FileDataset
        """
        if self.dcm_obj is not None:
            self._pull_info(self.dcm_obj)
        else:
            ds = pydicom.dcmread(self.filepath, stop_before_pixels=True)
            self._pull_info(ds)


class Cohort(GroupUtils):
    """Group level for Cohort with child of Patient

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): Specifies if FrameOfRef points to
            Series (True) or Modality (False). Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Patient
        self._organize_by = 'PatientID'
        if self.files:
            self._multithread_digest()
        self.isodatetime = utils.current_datetime()


class Patient(GroupUtils):
    """Group level for Patient, specified by PatientUID with child of FrameOfRef

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): Specifies if Study points to
            Series (True) or Modality (False). Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = FrameOfRef
        self._organize_by = 'FrameOfRefUID'
        self._digest()


class FrameOfRef(GroupUtils):
    """Group level for FrameOfReference, specified by FrameOfReferenceUID
        with child of Study

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): Specifies if Study points to
            Series (True) or Modality (False). Defaults to False.
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.decon = Deconstruction(tree=self)
        if self.filter_by is not None and 'StructName' in self.filter_by:
            self._reconstruct = Reconstruction(filter_structs=self.filter_by['StructName'])
        else:
            self._reconstruct = Reconstruction()
        self._child_type = Study
        self._organize_by = 'StudyUID'
        self._digest()

    def add_name_filter(self, names):
        self._reconstruct = Reconstruction(filter_structs=names)

    def recon(self, in_memory: bool = False, path: str = None):
        return self._reconstruct(self, in_memory=in_memory, path=path)


class Study(GroupUtils):
    """Group level for Study, specified by StudyUID with child of
        Series or Modality

    Args:
        name (str): A string declaring the group name
        include_series (bool, optional): Specifies if Study points to
            Series (True) or Modality (False). Defaults to False.
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
    """Group level for Series, specified by SeriesUID

    Args:
        name (str): A string declaring the group name
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._child_type = Modality
        self._organize_by = 'Modality'
        self._digest()

    def recon(self):
        raise NotImplementedError('Cannot reconstruct from Modality')


class Modality(GroupUtils):
    """Group level for Modality, specified by Modality

    Args:
        name (str): A string declaring the group name

    Attributes:
        ct: ct files within group
        nm: nm files within group
        mr: mr files within group
        pet: pet files within group
        dose: dose files within group
        struct: struct files within group
    """
    ct = property(utils.mod_getter('CT'), utils.mod_setter('CT'))
    nm = property(utils.mod_getter('NM'), utils.mod_setter('NM'))
    mr = property(utils.mod_getter('MR'), utils.mod_setter('MR'))
    pt = property(utils.mod_getter('PT'), utils.mod_getter('PT'))
    pet = property(utils.mod_getter('PET'), utils.mod_setter('PET'))
    dose = property(utils.mod_getter('RTDOSE'), utils.mod_setter('RTDOSE'))
    struct = property(utils.mod_getter('RTSTRUCT'), utils.mod_setter('RTSTRUCT'))

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dirname = name
        self.dicoms_data = {}
        self.volumes_data = {}
        self._child_type = [DicomFile, ReconstructedVolume, ReconstructedFile]
        self._organize_by = 'Modality'
        self._digest()

    def __str__(self) -> str:
        middle = []
        if self.dicoms_data:
            for index, key in enumerate(self.dicoms_data):
                middle.append(f' {len(self.dicoms_data[key])} file(s)')
        if self.volumes_data:
            for index, key in enumerate(self.volumes_data):
                data = self.volumes_data[key]
                files = len([x for x in data if type(x) is ReconstructedFile])
                fullvols = len([x for x in data if type(x) is ReconstructedVolume])
                if files:
                    middle.append(f' {files} volume file(s)')
                if fullvols:
                    middle.append(f' {fullvols} volume(s)')
        output = ' [' + ','.join(middle) + ' ]'
        return output

    def _add_file(self, item: object) -> None:
        # item is dicomfile or ReconstructionVolume
        key = str(item[self._organize_by])
        if type(item) is DicomFile:
            data = self.dicoms_data
        elif isinstance(item, ReconstructedVolume) or isinstance(item, ReconstructedFile):
            data = self.volumes_data
        else:
            raise TypeError('Added item must be DicomFile, Reconstructed(Volume / File)')

        if self._filter_check(item):
            if key in data.keys():
                if item not in data[key]:
                    data[key].append(item)
            else:
                data.update({key: [item]})

    def _filter_check(self, dicomfile: 'DicomFile') -> bool:
        if self.filter_by is not None:
            if 'Modality' in self.filter_by:
                if dicomfile.Modality not in self.filter_by['Modality']:
                    return False
        return True

    def recon(self):
        raise NotImplementedError('Cannot reconstruct from Modality')

    @property
    def dicoms(self) -> list:
        return self.dicoms_data[self.dirname]

    def clear_dicoms(self) -> None:
        self.dicoms_data = {}

    @property
    def volumes(self) -> list:
        return self.volumes_data[self.dirname]

    def clear_volumes(self) -> None:
        self.volumes_data = {}


@dataclass
class DicomDateTime:
    """Pulls the relevant date and time info from the DICOM header

    Attributes:
        StudyDate: (0008,0020) DICOM Tag
        SeriesDate: (0008, 0021) DICOM Tag
        ContentDate: (0008, 0023) DICOM Tag
        AcquisitionDate: (0008, 0022) DICOM Tag
        StudyTime: (0008,0030) DICOM Tag
        SeriesTime: (0008, 0031) DICOM Tag
        ContentTime: (0008, 0033) DICOM Tag
        AcquisitionTime: (0008, 0032) DICOM Tag

    Methods:
        iso_format (str): Returns ISO 8601 format date time for group type
    """
    ds: pydicom.dataset = None

    def __post_init__(self):
        if self.ds:
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
            self.ds = None

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def from_dict(self, fields: dict) -> object:
        for name, value in fields.items():
            setattr(self, name, value)
        return self

    def isoformat(self, group: str) -> str:
        """Returns ISO 8601 format date time for group type

        Args:
            group (str): String representing group type

        Returns:
            str: ISO 8601 formatted date time string with milliseconds
        """
        # ISO 8601 formatted date time string
        date, time = (None, None)
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

        if not date:
            date = '19700101'
        if not time:
            time = '000000.000000'

        if '.' in time:
            dateobj = datetime.strptime(date+time, '%Y%m%d%H%M%S.%f')
        else:
            dateobj = datetime.strptime(date+time, '%Y%m%d%H%M%S')
        dateobj = dateobj.replace(microsecond=0)
        str_name = dateobj.isoformat().replace(':', '.')
        return str_name

    def export(self) -> dict:
        return vars(self)
