import os
import shutil
import pydicom
from anytree import NodeMixin
from anytree.iterators.levelorderiter import LevelOrderIter
from datetime import datetime
from pathlib import Path


def save_tree(tree: NodeMixin, path: str, prefix: str = 'group') -> None:
    """[saves copy of dicom files to a specified location, ordered
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
        if modtype in self.data:
            modlist.append(self.data[modtype])
        return sorted(modlist)
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
