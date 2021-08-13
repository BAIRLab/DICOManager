import xarray as xr
import numpy as np
import pydicom

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
    ipp: list = None
    flipped: bool = False
    multi_thick: bool = False
    # If I want to get fancy I could make the fields below dynamic, reading from another
    # list of fields, but that is poor form
    PatientID: set = field(default_factory=set)
    Modality: set = field(defualt_factory=set)
    SOPInstanceUID: set = field(default_factory=set)
    StudyInstanceUID: set = field(defualt_factory=set)
    SeriesInstanceUID: set = field(defualt_factory=set)
    FrameOfReferenceUID: set = field(defualt_factory=set)
    SeriesDescription: set = field(defualt_factory=set)
    StudyDescription: set = field(defualt_factory=set)

    def __post_init__(self):
        # TODO: INTEGRATE DATETIME
        # TODO: ACCOUNT FOR THIS!
        if 'RTDOSE' in self.dicoms:
            filepath = self.dicoms['RTDOSE'][0].filepath
            ds = pydicom.dcmread(filepath)
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePositionPatient
            self.dz = float(ds.SliceThickness)
        else:
            if 'CT' in self.dicoms:
                files = self.dicoms['CT']
            elif 'MR' in self.dicoms:
                files = self.dicoms['MR']

            ds = self._calc_n_slices(files)

        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = map(float, ds.PixelSpacing)
        self.position = ds.PatientPosition
        self.dicoms = None

    def _pull_header_info(self, dicom):
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
            elif hasattr(dicom, field):
                value = getattr(dicom, field)
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
        """[calculates the number of volume slices]

        Args:
            files ([DicomFile]): [A list of DicomFile objects]

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
            ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
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
        shape = [self.rows, self.cols, self.slices]
        voxel_size = [self.dx, self.dy, self.dz]
        lo = self.origin
        hi = self.origin + voxel_size * shape
        output = {'x_ipp': np.linspace(lo[0], hi[0], self.dx),
                  'y_ipp': np.linspace(lo[1], hi[1], self.dy),
                  'z_ipp': np.linspace(lo[2], hi[2], self.dz)
                  }
        return output

    def report_attrs(self):
        output = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if type(value) is set:
                value = list(value)
            output.update({field: value})
        return output


def create_dataarray(volume, dicom_files):
    volume_dimensions = VolumeDimensions(dicom_files)
    attrs = volume_dimensions.report_attrs()
    coords = volume_dimensions.report_coords()

    new_da = xr.DataArray(volume, coords=coords, dims=['x_ipp', 'y_ipp', 'z_ipp'])
    new_da.attrs = attrs

    return new_da


@xr.registered_dataset_accessor("dims")
class ImageDims:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._obj.attr

    @property
    def field_of_view(self):
        shape = np.array(self.shape)
        voxel_size = np.array(self.voxel_size)
        return shape * voxel_size

    @property
    def shape(self):
        return [self._obj.rows, self._obj.columns, self._obj.slices]

    @property
    def voxel_size(self):
        return [self._obj.dx, self._obj.dy, self._obj.dz]

    def coordrange(self):
        pts_x = self._obj.origin[0] + np.arange(self._obj.rows) * self._obj.dx
        pts_y = self._obj.origin[1] + np.arange(self._obj.cols) * self._obj.dy
        pts_z = self._obj.origin[2] + np.arange(self._obj.slices) * self._obj.dz
        if self._obj._flipped:
            pts_z = pts_z[..., ::-1]
        return [pts_x, pts_y, pts_z]

    def coordgrid(self):
        pts_x, pts_y, pts_z = self.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)

    def calc_z_index(self, loc):
        return int(round(abs((self._obj.origin[-1] - loc) / self._obj.dz)))

    def Mgrid(self, dcm_hdr):
        """
        Function
        ----------
        Given a DICOM CT image slice, returns an array of pixel coordinates

        Returns
        ----------
        numpy.ndarray
            A numpy array of shape Mx2 where M is the dcm.Rows x dcm.Cols,
            the number of (x, y) pairs representing coordinates of each pixel

        Notes
        ----------
        Computes M via DICOM Standard Equation C.7.6.2.1-1
            https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037
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


@xr.registered_dataset_accessor("resample_update")
class ResampledUpdate:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ratio: flaot):
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


@xr.registared_dataset_accessor("crop_update")
class CropUpdate:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, values: list):
        xlo, xhi, ylo, yhi, zlo, zhi = values.T.flatten()
        self._obj.rows = xhi - xlo
        self._obj.cols = yhi - ylo
        self._obj.slices = zhi - zlo


@xr.registered_dataset_accessor("io")
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

    def save(self, filepath=None):
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


@xr.register_dataset_accessor("struct")
class StructureAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getattr__(self, name):
        if name not in self._obj.struct_names:
            return None
        index = self._obj.struct_names.index(name)
        return self._obj.structs[index]

    def as_dict(self):
        if self._obj['struct']:
            output = {}
            for name, vol in zip(self._obj['struct'], self._obj.struct_names):
                output.update({name: vol})
            return output
        return None


@xr.register_dataset_accessor("ct")
class CTAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


@xr.register_dataset_accessor("mr")
class MRAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj


# Each volume will be stored as a DataArray
# CT, MR, NM, RTDOSE will be [X, Y, Z]
# RTSTRUCT will be [N, X, Y, Z] where N is structure name

class ReconstructedSeries:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
