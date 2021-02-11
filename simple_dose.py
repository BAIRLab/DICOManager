
from glob import glob
import os
from scipy.interpolate import RegularGridInterpolator


def dose(patient_path, raises=False):
    if patient_path[0] == "~":
        patient_path = os.path.expanduser('~') + patient_path[1:]

    if patient_path[-1] != "/":
        patient_path += "/"

    dose_files = glob.glob(patient_path + "RTDOSE/*.dcm")
    ct_files = glob.glob(patient_path + "CT/*.dcm")
    ct_files.sort()

    dose_dcm = pydicom.dcmread(dose_files[0])

    # Determines if coordinate system is positive or negative
    for ct_file in ct_files:
        ct_dcm = pydicom.dcmread(ct_file, stop_before_pixels=True)
        if ct_dcm.InstanceNumber == 1:
            z_0 = float(ct_dcm.ImagePositionPatient[-1])
        if ct_dcm.InstanceNumber == len(ct_files):
            z_1 = float(ct_dcm.ImagePositionPatient[-1])

    try:
        img_origin = np.array(
            [*ct_dcm.ImagePositionPatient[:2], min(z_0, z_1)])
    except UnboundLocalError:
        print("This patient has multiple CT image sets or missing files")
        return 0

    img_dims = (ct_dcm.Rows, ct_dcm.Columns, len(ct_files))
    ix, iy, iz = (*ct_dcm.PixelSpacing, ct_dcm.SliceThickness)

    dose_iso = np.array(dose_dcm.ImagePositionPatient)
    dose_dims = np.rollaxis(dose_dcm.pixel_array, 0, 3).shape
    dx, dy, dz = (*dose_dcm.PixelSpacing, dose_dcm.SliceThickness)

    d_grid_x = dose_iso[1] + dx * np.arange(dose_dims[0])
    d_grid_y = dose_iso[0] + dy * np.arange(dose_dims[1])
    d_grid_z = dose_iso[2] + dz * np.arange(dose_dims[2])

    i_grid_x = img_origin[1] + ix * np.arange(img_dims[0])
    i_grid_y = img_origin[0] + iy * np.arange(img_dims[1])
    i_grid_z = img_origin[2] + iz * np.arange(img_dims[2])

    grids = [(i_grid_x, d_grid_x), (i_grid_y, d_grid_y), (i_grid_z, d_grid_z)]
    list_of_grids = []

    def nearest(array, value):
        return np.abs(np.asarray(array) - value).argmin()

    # Compute the nearest CT coordinate to each dose coordinate
    for _, (img_grid, dose_grid) in enumerate(grids):
        temp = list(set([nearest(img_grid, val) for val in dose_grid]))
        temp.sort()
        list_of_grids.append(np.array(temp))

    dose_array = np.rollaxis(dose_dcm.pixel_array, 0, 3)

    x_mm, y_mm, z_mm = [[t[0], t[-1]] for t in list_of_grids]
    total_pts = np.product([[y - x] for x, y in [x_mm, y_mm, z_mm]])
    grid = np.mgrid[x_mm[0]: x_mm[1],
                    y_mm[0]: y_mm[1],
                    z_mm[0]: z_mm[1]].reshape(3, total_pts)
    interp_pts = np.squeeze(np.array([grid]).T)

    interp = RegularGridInterpolator(
        list_of_grids, dose_array, method='linear')
    interp_vals = interp(interp_pts)
    interp_vol = interp_vals.reshape(x_mm[1] - x_mm[0],
                                     y_mm[1] - y_mm[0],
                                     z_mm[1] - z_mm[0])
    full_vol = np.zeros(img_dims)

    fill_volume[x_mm[0]: x_mm[1],
                y_mm[0]: y_mm[1],
                z_mm[0]: z_mm[1]] = interp_vol

    return full_vol * float(dose_dcm.DoseGridScaling)
