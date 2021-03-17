from skimage.metrics import structural_similarity as ssim
import pydicom
from glob import glob
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI


# Need the DicomGroup to have:
# - iterable through the files
# - property for the number of associated files
# - has a modality

# This only works for CT currently, could adapt to work with Dose too...

# series.ct will give [DicomFile objects]
@dataclass
class VolumeDimensions:
    dicoms: list
    modality: str = dicoms[0].Modality
    origin: list = None
    rows: int = None
    cols: int = None
    slices: int = None
    dims: list = None
    dx: float = None
    dy: float = None
    dz: float = None
    flipped: bool = False

    def __post_init__(self):
        if self.modality == 'RTDOSE':
            ds = pydicom.dcmread(self.dicoms[0].filepath)
            self.slices = ds.NumberOfFrames
            self.origin = ds.ImagePatientPosition
        else:
            for dcm in self.dicoms:
                ds = pydicom.dcmread(dcm.filepath, stop_before_pixels=True)
                ipp = ds.ImagePatientPosition
                if ds.InstanceNumber == 1:
                    self._z0 = float(ipp[-1])
                if ds.InstanceNumber == self.dicoms.nfiles:
                    self._z1 = float(ipp[-1])

            self.origin = np.array([*ipp[:2], min(self._z0, self._z1)])
            self._tot_z = max(self._z0, self._z1) - min(self._z0, self._z1)
            self.slices = self._tot_z / self.dz
            if self._z1 > self._z0:
                self.flipped = True
        
        self.rows = ds.Rows
        self.cols = ds.Columns
        self.dx, self.dy = ds.PixelSpacing
        self.dz = ds.SliceThickness
    
    @property
    def shape(self):
        return (self.rows, self.cols, self.slices)

    def coordrange(self):
        pts_x = self.origin[0] + np.arange(self.rows) * self.dx
        pts_y = self.origin[1] + np.arange(self.cols) * self.dy
        pts_z = self.origin[2] + np.arange(self.slices) * self.dz
        if self.flipped:
            pts_z = pts_z[..., ::-1]
        return (pts_x, pts_y, pts_z)

    def coordgrid(self):
        pts_x, pts_y, pts_z = self.coordrange()
        grid = np.array([*np.meshgrid(pts_x, pts_y, pts_z, indexing='ij')])
        return grid.reshape(3, -1)


# tools
def dose_max_point(dose_array):
    idx = np.unravel_index(np.argmax(dose_array), dose_array.shape)
    return idx


def dose_interp(Series):
    volumes = {}
    ct_dims = VolumeDimensions(Series.ct)
    ct_coords = ct_dims.coordgrid()
    for dosefile in Series.dose:
        ds = pydicom.dcmread(dosefile.filepath)
        dose_dims = VolumeDimensions(dosefile)
        dose_array = np.rollaxis(ds.pixel_array, 0, 3) * ds.DoseGridScaling
        dose_coords = dose_dims.coordrange()
        interper = RGI(dose_coords, dose_array, bounds_error=False, fill_value=0)
        dose_interp = interper(ct_coords).reshape(ct_dims.shape)
        volumes.update({dosefile: dose_interp})
    return volumes
