import io
import pickle
import uuid
import hashlib
import sys
from pathlib import Path
from . import utils


class RestrictedUnpickler(pickle.Unpickler):
    """An overriding of pickle loading to only allow ProtectedFile type
    """
    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if module == '__main__' or module == 'DICOManager.dcmio':
            if name == 'ProtectedFile':
                return getattr(sys.modules[module], name)

        # Forbid everything else.
        message = f'{module}.{name} is not supported, this file may have been tampered with'
        raise pickle.UnpicklingError(message)


class ProtectedFile:
    """A file with an SHA256 and MAC address metadata

    WARNING:
        While no pickle object can be entirely safe, short of encryption,
        the user barrier when using encryption is high. Therefore this is not a
        fool-proof protection scheme and is not a replacement for local disk
        encryption, proper read-write protections and not loading files from
        untrusted sources. Please use discretion.
    """
    def __init__(self, obj):
        self.pickled_object = pickle.dumps(obj)
        self.mac = uuid.getnode()
        self.sha = hashlib.sha256(self.pickled_object).hexdigest()


def dump_tree(tree: object, filepath: str) -> None:
    """Dumps tree to a given filepath location

    Args:
        tree (object): DICOManager object to save to disk
        filepath (str): POSIX directory to write file

    Notes:
        This function does not create directories if not otherwise present.
    """
    # modify to use path, checking if filename is specified, otherwise create filename
    path = Path(filepath)

    if path.is_dir():
        path = path / tree.name

    path = path.parent / (path.name + ".dcmpkl")

    with open(path, 'wb') as fname:
        pickle.dump(ProtectedFile(tree), fname)

    print(f'{tree.name} written to: {path}')


def load_tree(filename: str, force_mac: bool = False, force_sha: bool = False) -> object:
    """Load a DICOManager pickle object (.dcmpkl). By default, this will only load objects
        created on the local machine that are unchanged since their save state. This is to
        prevent the loading and execution of unsafe pickle objects created by other users.
        If the code is run on a scheduled compute array, the MAC address of the system
        executing the code will likely differ each submission. Therefore force_mac=True may
        be required.

    Args:
        filename (str): POSIX path and filename
        force_mac (bool, optional): Overrides MAC address check. Defaults to False.
        force_sha (bool, optional): Overrides SHA256 check. Defaults to False.

    WARNING:
        While no pickle object can be entirely safe, short of encryption,
        the user barrier when using encryption is high. Therefore this is not a
        fool-proof protection scheme and is not a replacement for local disk
        encryption, proper read-write protections and not loading files from
        untrusted sources. Please use discretion.
    """
    def _load_protected_file(group):
        """Helper function analogous to pickle.loads()."""
        return RestrictedUnpickler(io.BytesIO(group)).load()

    if force_sha:
        utils.colorwarn('Use force_sha=True with caution, unsafe code may be executed')

    with open(filename, 'rb') as fname:
        fbytes = fname.read()
        save_file = _load_protected_file(fbytes)
        sha = hashlib.sha256(save_file.pickled_object).hexdigest()
        mac = uuid.getnode()
        sha_condition = force_sha or sha == save_file.sha
        mac_condition = force_mac or mac == save_file.mac

        if sha != save_file.sha:
            utils.colorwarn('Tree ojbect modified since saving, force_sha=True NOT recommened')
        if mac != save_file.mac:
            message1 = 'Tree object created on a different system, '
            message2 = 'use force=True to load at your own risk'
            utils.colorwarn(message1 + message2)

        if sha_condition and mac_condition:
            return pickle.loads(save_file.pickled_object)
