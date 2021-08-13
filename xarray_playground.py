import xarray as xr
import numpy as np

def create_dataarray(volume, dcm):
    x_lo, y_lo, z_lo = dcm.ImagePositionPatient
    dx, dy, dz = [dcm.PixelSpacing, dcm.SliceThickness]
    nx, ny, nz = [dcm.Columns, dcm.Rows, dcm.Slices]
    x_hi = x_lo + dx*nx
    y_hi = y_lo + dy*ny
    z_hi = z_lo + dz*nz

    coords = {'x_ipp': np.linspace(x_lo, x_hi, nx),
              'y_ipp': np.linspace(y_lo, y_hi, ny),
              'z_ipp': np.linspace(z_lo, z_hi, nz),
              'dx': dx,
              'dy': dy,
              'dz': dz,
              'columns': nx,
              'rows': ny,
              'slices': nz
              }

    attr_fields = ['PatientID', 'StudyUID', 'SeriesUID', 'FrameOfReferenceUID',
                   'StudyDescription', 'SeriesDescription', 'Modality', 'SOPInstanceUID']
    attrs = {}
    for field in attr_fields:
        attrs.update({field: dcm.getattr(field)})

    new_da = xr.DataArray(volume, coords=coords, dims=['x_ipp', 'y_ipp', 'z_ipp'])
    new_da.attrs = attrs

    return new_da


@xr.registered_dataset_accessor("_backend")
class Backend:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._files = self._obj._files
        self._obj.slices = self._calc_n_slices()
        self._obj.dz = self._calculate_dz()
        del self._obj._files

    def _calculate_n_slices(self):
        return 1

    def _calculate_dz(self):
        return 1


@xr.registered_dataset_accessor("dims")
class ImageDims:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._obj.attrs = {'NewlyComputedAttrsGoHere': 1}

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
