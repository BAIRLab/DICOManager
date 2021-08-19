import xarray as xr
import numpy as np
import pydicom
from dataclasses import dataclass, field
from glob import glob

# Can either create data array with function, or feed the dcm_header and volume
# into data array and use the backend accessor to compute those things.
# I'm leaning towards using the create function method to save from extra baggage
# in the xarray objects

@dataclass
class VolumeDimensions:
    dicoms: list
    origin: list = None
    rows: int = None
    cols: int = None
    slices: int = None
    dx: float = None
    dy: float = None
    dz: float = None
    voxel_size: list = None
    ipp: list = None
    flipped: bool = False
    multi_thick: bool = False
    # If I want to get fancy I could make the fields below dynamic, reading from another
    # list of fields, but that is poor form
    PatientID: set = field(default_factory=set)
    Modality: set = field(default_factory=set)
    SOPInstanceUID: set = field(default_factory=set)
    StudyInstanceUID: set = field(default_factory=set)
    SeriesInstanceUID: set = field(default_factory=set)
    FrameOfReferenceUID: set = field(default_factory=set)
    SeriesDescription: set = field(default_factory=set)
    StudyDescription: set = field(default_factory=set)

    def __post_init__(self):
        # TODO: INTEGRATE DATETIME
        # TODO: ACCOUNT FOR THIS!
        ds = self._calc_n_slices(self.dicoms)
        if ds.Modality == 'RTDOSE':
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePositionPatient
            self.dz = float(ds.SliceThickness)

        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = map(float, ds.PixelSpacing)
        self.voxel_size = np.array([self.dx, self.dy, self.dz], dtype=np.half)
        self.position = ds.PatientPosition

    def _pull_header_info(self, ds):
        for field in self.__dataclass_fields__:
            if field == 'FrameOfReferenceUID':
                if hasattr(ds, 'FrameOfReferenceUID'):
                    frame = ds.FrameOfReferenceUID
                elif hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
                    ref_seq = ds.ReferencedFrameOfReferenceSequence
                    frame = ref_seq[0].FrameOfReferenceUID
                else:
                    frame = None
                self.FrameOfReferenceUID = frame
            elif hasattr(ds, field):
                value = getattr(ds, field)
                setattr(self, field, value)

    def _calculate_dz(self, zlocations):
        zlocations.sort()
        differences = np.zeros((len(zlocations) - 1))
        previous = min(zlocations)

        for i in range(1, len(zlocations)):
            differences[i - 1] = round(abs(previous - zlocations[i]), 2)
            previous = zlocations[i]

        differences = list(set(differences))

        if len(differences) > 1:
            self.multi_thick = True
        return min(differences)

    def _calc_n_slices(self, files: list):
        """Calculates the number of volume slices

        Args:
            files (DicomFile): A list of DicomFile objects

        Notes:
            Creating the volume by the difference in slice location at high and
            low instances ensures proper registration to rstructs, even if
            images slices are missing. We can interpolate to the lowest
            instance if we do not have instance 1, but we cannot extrapolate
            if higher instances are missing
        """
        z0, z1 = (np.inf, -np.inf)
        low_inst = np.inf
        low_thickness = None
        header_thicknesses = []
        zlocations = []

        for dcm in files:
            ds = pydicom.dcmread(dcm, stop_before_pixels=True)
            if ds.Modality == 'RTDOSE':
                return ds
            self._pull_header_info(ds)
            self.ipp = ds.ImagePositionPatient
            self.iop = ds.ImageOrientationPatient
            header_thicknesses.append(float(ds.SliceThickness))
            zloc = float(self.ipp[-1])
            zlocations.append(zloc)

            if zloc < z0:
                z0 = zloc
            if zloc > z1:
                z1 = zloc

            if hasattr(ds, 'InstanceNumber'):
                inst = int(ds.InstanceNumber)
                if inst < low_inst:
                    low_inst = inst
                    low_thickness = ds.SliceThickness

        if 1 < low_inst < 5 and low_inst != np.inf:  # For extrapolation of missing slices
            # Need to make it the slice thickness of the lowest slice
            # in case the image has mixed thicknesses
            z0 -= low_thickness * (low_inst - 1)
            low_inst = 1

        self.dz = self._calculate_dz(zlocations)
        self.zlohi = (z0, z1)
        self.origin = np.array([*self.ipp[:2], max(z0, z1)])
        self.slices = 1 + round((max(z0, z1) - min(z0, z1)) / self.dz)

        return ds

    def report_coords(self):
        output = {'x_ipp': self.origin[0] + np.arange(self.rows) * self.dx,
                  'y_ipp': self.origin[1] + np.arange(self.cols) * self.dy,
                  'z_ipp': self.origin[2] - np.arange(self.slices) * self.dz
                  }
        for key, value in output.items():
            output[key] = np.array(value, dtype=np.half)
        return output

    def report_attrs(self):
        output = {}
        for field in self.__dataclass_fields__:
            if field == 'dicoms':
                continue
            value = getattr(self, field)
            if type(value) is set:
                value = list(value)
            output.update({field: value})
        return output



@xr.register_dataarray_accessor("is_struct")
class IsStruct:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        return self._obj.Modality == 'RTSTRUCT'


@xr.register_dataarray_accessor("field_of_view")
class VolumeShape:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        shape = np.array(self._obj.shape)
        voxel_size = np.array(self._obj.voxel_size)
        if self._obj.is_struct:
            return shape[1:] * voxel_size
        return shape * voxel_size


@xr.register_dataarray_accessor("coordrange")
class CoordRange:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        pts_x = self._obj.origin[0] + np.arange(self._obj.rows) * self._obj.dx
        pts_y = self._obj.origin[1] + np.arange(self._obj.cols) * self._obj.dy
        pts_z = self._obj.origin[2] + np.arange(self._obj.slices) * self._obj.dz
        if self._obj._flipped:
            pts_z = pts_z[..., ::-1]
        return [pts_x, pts_y, pts_z]


@xr.register_dataarray_accessor("coordgrid")
class CoordGrid:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        pts_x, pts_y, pts_z = self._obj.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)


@xr.register_dataarray_accessor("calc_z_index")
class CalcZIndex:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        return int(round(abs((self._obj.origin[-1] - loc) / self._obj.dz)))


@xr.register_dataarray_accessor("mgrid")
class MGrid:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        """Given a DICOM CT image slice, returns an array of pixel coordinates
           for the given frame of reference. Useful for converting boolean masks
           to RTSTRUCT files

        Args:
            dcm_hdr (pydicom.dataset.Dataset): pydicom dataset object of DICOM header

        Returns:
            np.ndarray: A numpy array of the patient position coordinates
        """
        IOP = dcm_hdr.ImageOrientationPatient
        IPP = dcm_hdr.ImagePositionPatient
        # Unpacking arrays is poor form, but I'm feeling rebellious...
        X_x, X_y, X_z = np.array(IOP[:3]).T
        Y_x, Y_y, Y_z = np.array(IOP[3:]).T
        S_x, S_y, S_z = np.array(IPP)
        D_i, D_j = self._obj.dx, self._obj.dy
        i, j = np.indices((self._obj.rows, self._obj.cols))

        M = np.array([[X_x*D_i, Y_x*D_j, 0, S_x],
                      [X_y*D_i, Y_y*D_j, 0, S_y],
                      [X_z*D_i, Y_z*D_j, 0, S_z],
                      [0, 0, 0, 1]])
        C = np.array([i, j, np.zeros_like(i), np.ones_like(i)])

        return np.rollaxis(np.stack(np.matmul(M, C)), 0, 3)


@xr.register_dataarray_accessor("_resample_update")
class ResampledUpdateArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ratio: float):
        if type(ratio) is int or type(ratio) is float:
            ratio = np.array([ratio, ratio, ratio])
        if type(ratio) is list:
            ratio = np.array(ratio)

        self._obj.dx /= ratio[0]
        self._obj.dy /= ratio[1]
        self._obj.dz /= ratio[2]

        self._obj.rows = int(round(self.rows * ratio[0]))
        self._obj.cols = int(round(self.rows * ratio[1]))
        self._obj.slices = int(round(self.slices * ratio[2]))


@xr.register_dataarray_accessor("_crop_update")
class CropUpdateArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, values: list):
        xlo, xhi, ylo, yhi, zlo, zhi = values.T.flatten()
        self._obj.rows = xhi - xlo
        self._obj.cols = yhi - ylo
        self._obj.slices = zhi - zlo


@xr.register_dataset_accessor('iter_modalities')
class IterModalities:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        for mod in ds.data_vars:
            yield mod


@xr.register_dataset_accessor("_resample_update")
class ResampledUpdateSet:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ratio: float):
        for mod in self._obj.iter_modalities():
            mod._resample_update(ratio)


@xr.register_dataset_accessor("_crop_update")
class CropUpdateSet:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, values: list):
        for mod in self._obj.iter_modalities():
            mod._crop_update(values)


# We could make each of theses methods standalone functions, but I'm
# afraid that basic ones like 'save' and 'load' will interfere or overload
# the default xarray functions. Therefore I've offset them in the io category
@xr.register_dataset_accessor("io")
class ReconstructedIO:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._is_pointer = False
        self._filepath = None

    def __str__(self):
        if self.is_pointer():
            return self
        else:
            return self

    def _generate_filepath(self):
        self._filepath = 'NewFilepath'

    def _rename(self):
        pass

    def _load(self):
        temp = xr.load(self._filepath)
        self._obj.data = temp.data

    def _write(self, filepath):
        xr.save(self._obj, filepath)

    @property
    def is_pointer(self):
        return self._is_pointer()

    def save_array(self, filepath=None):
        if not self.is_pointer():
            if filepath:
                self._filepath = filepath
            elif not self._filepath():
                self._generate_filepath()
            self._write(self._filepath)
        else:
            # Warn type is pointer
            pass

    def load_array(self):
        if self.is_pointer():
            # would load from filepath
            self._is_pointer = False
            self._load()

    def convert_to_pointer(self):
        if not self.is_pointer():
            # would write to filepath
            self.save_file()
            self._obj.data = None
            self._is_pointer = True


def create_dataset(Modality):
    dims = {}
    for mod, dicoms in Modality.dicom_files:
        dims.update({mod: create_dataarray(dicoms)})
    new_ds = xr.Dataset(dims)
    Modality.volume_data = new_ds


def create_dataarray(volume, dicom_files, struct_names=None):
    volume_dimensions = VolumeDimensions(dicom_files)
    attrs = volume_dimensions.report_attrs()
    coords = volume_dimensions.report_coords()
    name = volume_dimensions.SeriesInstanceUID

    dims = ['x_ipp', 'y_ipp', 'z_ipp']
    if struct_names is not None:
        dims = ['struct_names'] + dims
        new_coords = ({'struct_names': struct_names})
        new_coords.update(coords)
        coords = new_coords

    new_da = xr.DataArray(volume, name=name, coords=coords)#, dims=dims)
    new_da.attrs = attrs

    return new_da


# Each image will be a xarray.DataArray
# The dims will simply be the x, y, z coordinates
# For a structure, there will also be structure names
# structure will have a hidden accessor for is_structure
# Then the overall reconstructed group will be stored as a
# xarray.Dataset with the same coordinates for x, y, z
# and a series of special methods to implement special group
# funciontality

if __name__ == '__main__':
    ct_files = glob('/home/eporter/eporter_data/rtog_project/dicoms/train/**/Patient_10*71/**/CT/*.dcm', recursive=True)
    volume = np.zeros((5, 512, 512, len(ct_files)))
    print(volume.shape)
    struct_names = ['rt' + str(n) for n in range(5)]
    xrda = create_dataarray(volume=volume, dicom_files=ct_files, struct_names=struct_names)
    xrds = xrda.to_dataset()
    print(xrda)
    print(xrda.attrs)
    print(xrda.field_of_view())
    print(xrds)