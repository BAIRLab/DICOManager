#!/usr/bin/python3
import collections
import glob
import os
import numpy as np
import pydicom
import cv2
import utils
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


__author__ = ["Evan Porter", "David Solis", "Ron Levitin"]
__license__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


def mri(patient_path, path_mod=False, raises=False):
    """
    Function
    ----------
    Reconstructs a MRI volume

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    path_mod : str (Default = None)
        Specifies a suffix of the CT directory, such as CT0, CT1, ...
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    Index Error
        Raised if instances are missing from the MRI directory

    Returns
    ----------
    mr_array : np.array
        A reconstructed MRI array in the shape of [x, y, z]
    """

    # Check path type
    # -- if dicom file, check directory for related dicoms
    # -- if directory, check for "[CT, MRI, CBCT, PET, RTSTRUCT, RTDOSE]" folder and read those in.

    # If path is a string, convert to pathlib.Path
    if not isinstance(patient_path, Path):
        patient_path = Path(patient_path).expanduser()

    # If directly given a dicom
    if patient_path.is_file():
        volume_slices = utils.find_series_slices(patient_path)
        dcmheader = pydicom.dcmread(volume_slices[0])
    elif patient_path.is_dir():
        patient_path = patient_path / "MR"
        if patient_path.is_dir():
            volume_slices = utils.find_series_slices(patient_path)
            dcmheader = pydicom.dcmread(volume_slices[0])
        else:
            err_msg = f"No MR folder in patient path: {patient_path.parent}"
            if raises:
                raise ValueError(err_msg)
            else:
                print(err_msg)

    slice_thick, n_z, loc0, _, flip, _ = utils.img_dims(volume_slices)
    image_array = np.zeros((*dcmheader.pixel_array.shape, n_z), dtype="float32")

    try:
        for slice_file in volume_slices:
            ds = pydicom.dcmread(str(slice_file))
            try:
                z_loc = int(round(abs((loc0-ds.SliceLocation) / slice_thick)))
            except Exception:
                ipp = ds.ImagePositionPatient
                z_loc = int(round(abs((loc0-ipp[-1]) / slice_thick)))
            image_array[:, :, z_loc] = ds.pixel_array

    except IndexError:
        if raises:
            raise IndexError(f"There is a discontinuity in {patient_path}")
        else:
            print(f"This is a discontinuity in {patient_path}")
    else:
        if not flip:
            image_array = image_array[..., ::-1]
    return image_array


def struct(patient_path, wanted_contours, raises=False):
    """
    Function
    ----------
    Reconstructs a RTSTRUCT volume(s)

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    contour_names : list or dict
        A list of contour names that will be included if found. Array will order
            in the same format as the received list of names, or order of keys
        If list, then only items in the list will be output. If a dict, then
            each list will map to the key and saved the respective index
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    ValueError
        Raised if the RTSTRUCT and CT are not registered
    ValueError
        Raised if values are missing from an RTSTRUCT contour sequence

    Returns
    ----------
    rtstruct_array : np.array
        A reconstructed RTSTRUCT array in the shape of [struct, x, y, z], where
            x, y, z are the same dimensions as the registered CT volume

    Notes
    ----------
    MIM saves the contour data at a sampling rate twice that of the image coordinate
        system. This function resamples the contour to the image coordinate system.
        If high precision is required, the following changes should be made:
            dimensions = (2 * ..., 2 * ..., vol_n_z)
            ix, iy, iz = (*volume_dcm.PixelSpacing / 2, ...)
        Which should generate a mask of dimensions 1024x1024xN for a 512x512xN image
    """
    # If path is a string, convert to pathlib.Path
    if not isinstance(patient_path, Path):
        patient_path = Path(patient_path).expanduser()

    # find the RTSTRUCT dcm file
    if patient_path.is_file() and patient_path.suffix == ".dcm":
        struct_file = patient_path
    elif patient_path.is_dir():
        patient_path_sub = patient_path / "RTSTRUCT"
        if patient_path_sub.is_dir():
            if len(list(patient_path_sub.iterdir())) > 1:
                struct_file = sorted(patient_path_sub.iterdir(), key=utils.struct_sort, reverse=True)[0]
            else:
                struct_file = next(patient_path_sub.iterdir())
        else:
            err_msg = f"No RTSTRUCT folder in patient path: {patient_path.parent}"
            if raises:
                raise ValueError(err_msg)
            else:
                print(err_msg)

    # Open RTSTRUCT Header
    struct_dcm = pydicom.dcmread(str(struct_file))
    if not struct_dcm.Modality == "RTSTRUCT":
        err_msg = (
            f"DICOM header for file {struct_file} is of Modality {struct_dcm.Modality}"
        )
        raise ValueError(err_msg) if raises else print(err_msg)

    # Find the associated volume dicoms and read them.
    # Need the associated volume for the image space information
    volume_slice_files = utils.find_series_slices(str(struct_file), find_associated=True)
    for vfile in volume_slice_files:
        volume_dcm = pydicom.dcmread(vfile, stop_before_pixels=True)
        try:
            if volume_dcm.InstanceNumber == 1:
                break
        except Exception:
            print(volume_dcm)

    _, vol_n_z, _, _, flip, _ = utils.img_dims(volume_slice_files)
    dimensions = (volume_dcm.Rows, volume_dcm.Columns, vol_n_z)
    img_origin = np.array(volume_dcm.ImagePositionPatient)
    ix, iy, iz = (*volume_dcm.PixelSpacing, volume_dcm.SliceThickness)

    # inner function to convert the points to coords
    def _points_to_coords(contour_data, img_origin, ix, iy, iz):
        points = np.array(
            np.round(abs(contour_data - img_origin) / [ix, iy, abs(iz)]), dtype=np.int32)

        return points

    # This function requires a list of the contours being looked for. Can be dict or list
    if type(wanted_contours) is dict:
        wanted_contours = dict((k.lower(), v) for k, v in wanted_contours.items())
    else:
        wanted_contours = [item.lower() for item in wanted_contours]

    masks = []
    contours = []  # Designates the contours that have been included
    for index, contour in enumerate(struct_dcm.StructureSetROISequence):
        # Functionality for dictionaries
        if type(wanted_contours) is dict:
            for key in wanted_contours:
                if contour.ROIName.lower() in wanted_contours[key]:
                    contour.ROIName = key
                    continue
        if contour.ROIName.lower() not in wanted_contours:
            continue

        contours.append(contour.ROIName.lower())

        fill_array = np.zeros(shape=dimensions)
        if hasattr(struct_dcm.ROIContourSequence[index], 'ContourSequence'):
            contour_list = struct_dcm.ROIContourSequence[index].ContourSequence

            for contour_slice in contour_list:
                try:
                    contour_data = np.array(contour_slice.ContourData).reshape(-1, 3)
                except ValueError:
                    err_msg = f"Contour {contour.ROIName} in {struct_file} is corrupt"
                    raise ValueError(err_msg) if raises else print(err_msg)

                points = _points_to_coords(
                    contour_data, img_origin, ix, iy, iz)
                coords = np.array([points[:, :2]], dtype=np.int32)
                # scimage.draw.Polygon is incorrect, use cv2.fillPoly instead
                poly_2D = np.zeros(dimensions[:2])
                cv2.fillPoly(poly_2D, coords, 1)
                fill_array[:, :, points[0, 2]] += poly_2D

        # Protect against any overlaps in the contour
        fill_array = fill_array % 2
        if flip:
            fill_array = fill_array[..., ::-1]
        masks.append(fill_array)
    # Reorders the list to match the wanted contours
    key_list = utils.key_list_creator(wanted_contours)
    ordered = [masks[contours.index(x)]
               for x in sorted(contours, key=key_list)]
    return np.array(ordered, dtype='bool'), contours


def nm(patient_path, raises=False):
    """
    Function
    ----------
    Reconstructs a NM volume

    Parameters
    ----------
    patient_pat : str
        A path directing towards a patient database in the following format:
            MRN/NM/*.dcm
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    AssertionError
        Raised if the directory contains more than 1 NM volume

    Returns
    ----------
    numpy.ndarray
        A [X, Y, Z] dimension numpy array of the NM volume

    Notes
    ----------
    The values are returned raw (as they are stored). If the DICOM header specifies
        offsets or slope adjustments, that is not included currently
    Each volume appears to be flipped in the z-axis, so it will be flipped to be HFS
        but, this may not be entirely robust and the flip param in img_dims may be
        needed here to match the CT flipping
    """
    if patient_path[-1] != '/':
        patient_path += '/'

    nm_files = glob.glob(patient_path + 'NM/*.dcm')
    nm_files.sort()

    if raises:
        assert len(nm_files) > 1, 'can only reconstruct one nm file at a time'
    elif len(nm_files) > 1:
        print('Will only construct the first nm file in this dir')

    ds = pydicom.dcmread(nm_files[0])
    # DICOM stores as [z, x, y]
    raw = np.rollaxis(ds.pixel_array, 0, 3)[:, :, ::-1]
    # NM Rescale slope and intercept values
    slope = ds.RealWorldValueMappingSequence[0].RealWorldValueSlope
    intercept = ds.RealWorldValueMappingSequence[0].RealWorldValueIntercept
    return raw * slope + intercept


def ct(patient_path, path_mod=None, HU=False, raises=False):
    """
    Function
    ----------
    Reconstructs a CT volume (Should feasibly work on MR as well, but untested)

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    path_mod : str (Default = None)
        Specifies a suffix of the CT directory, such as CT0, CT1, ...
    HU : bool (Default = False)
        Converts the CT image array into HU values
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    Index Error
        Raised if instances are missing from the CT directory

    Returns
    ----------
    ct_array : np.array
        A reconstructed CT array in the shape of [x, y, z]
    """
    if patient_path[0] == "~":
        patient_path = os.path.expanduser("~") + patient_path[1:]

    if patient_path[-1] != "/":
        patient_path += "/"

    ct_files = glob.glob(patient_path + "CT/*.dcm")
    ct_files.sort()
    ct_dcm = pydicom.dcmread(ct_files[0])

    ct_thick, ct_n_z, ct_loc0, ct_loc1, flip, mult_thick = utils.img_dims(ct_files)
    ct_array = np.zeros((*ct_dcm.pixel_array.shape,
                         ct_n_z), dtype='float32')

    try:
        for ct_file in ct_files:
            ds = pydicom.dcmread(ct_file)
            z_loc = int(round(abs((ct_loc0-ds.SliceLocation) / ct_thick)))
            ct_array[:, :, z_loc] = ds.pixel_array

    except IndexError:
        if raises:
            raise IndexError(f"There is a discontinuity in {patient_path}CT")
        else:
            print(f"This is a discontinuity in {patient_path}CT")
    else:
        if flip:
            ct_array = ct_array[..., ::-1]
        if mult_thick:
            print(f"Multiple slice thicknesses found in {patient_path}CT")
            ct_array = utils.multi_slice_resample(ct_array)
        if HU:
            return ct_array * ct_dcm.RescaleSlope + ct_dcm.RescaleIntercept
        else:
            return ct_array


def pet(patient_path, path_mod=None, raises=False):
    """
    Function
    ----------
    Reconstructs a PET volume

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    path_mod : str (Default = None)
        Specifies a suffix of the PET directory, such as PET0, PET1, ...
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    ValueError
        Raised if the PET and CT are not registered
    IndexError
        Raised if any instances are missing from the PET directory

    Returns
    ----------
    pet_array : np.array
        A reconstructed PET image in the same dimensions as the registered CT
    """
    if patient_path[0] == "~":
        patient_path = os.path.expanduser("~") + patient_path[1:]

    if patient_path[-1] != "/":
        patient_path += "/"

    if os.path.isdir(patient_path + "CT" + str(path_mod)):
        ct_files = glob.glob(patient_path + "CT" + str(path_mod) + "/*.dcm")
    else:
        ct_files = glob.glob(patient_path + "CT/*.dcm")

    ct_dcm = pydicom.dcmread(ct_files[0], stop_before_pixels=True)

    pet_files = glob.glob(patient_path + "PET" + str(path_mod) + "/*.dcm")
    pet_dcm = pydicom.dcmread(pet_files[0])

    pet_thick, pet_n_z, pet_loc0, pet_loc1, flip, _ = utils.img_dims(pet_files)
    pet_array = np.zeros((*pet_dcm.pixel_array.shape,
                          pet_n_z), dtype='float32')

    if pet_dcm.FrameOfReferenceUID != ct_dcm.FrameOfReferenceUID:
        if raises:
            raise ValueError(
                f"These two images are not registered: {ct_files[0]} & {pet_files[0]}"
            )
        else:
            print(
                f"These two images are not registered: {ct_files[0]} & {pet_files[0]}"
            )
    else:
        try:
            for _, pet_file in enumerate(pet_files):
                ds = pydicom.dcmread(pet_file)
                z_loc = int(round(abs((pet_loc0 - ds.SliceLocation) / pet_thick)))
                pet_array[:, :, z_loc] = ds.pixel_array
        except IndexError:
            if raises:
                raise IndexError("There is a discontinuity in your images")
            else:
                print(f"There is a discontinuity in {patient_path}PET")
        else:
            if flip:
                pet_array = pet_array[..., ::-1]
            rescaled = pet_array * pet_dcm.RescaleSlope + pet_dcm.RescaleIntercept
            patient_weight = 1000 * float(pet_dcm.PatientWeight)
            total_dose = float(
                pet_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            )
            return rescaled * patient_weight / total_dose


def dose(patient_path, raises=False):
    """
    Function
    ----------
    Reconstructs a RTDOSE volume

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    Value Error
        Raised if instances are missing from the CT directory

    Returns
    ----------
    ct_array : np.array
        A reconstructed CT array in the shape of [x, y, z]
    """
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
        img_origin = np.array([*ct_dcm.ImagePositionPatient[:2], min(z_0, z_1)])
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

    padding = 25  # Padding since some RTDOSE extend outside the CT volume
    i_grid_x = img_origin[1] + ix * np.arange(img_dims[0])
    i_grid_y = img_origin[0] + iy * np.arange(img_dims[1])
    i_grid_z = img_origin[2] + iz * np.arange(-padding, img_dims[2] + padding)
    i_grid_z = i_grid_z[::-1]  # Likely redundant with lower flips

    grids = [(i_grid_x, d_grid_x), (i_grid_y, d_grid_y), (i_grid_z, d_grid_z)]
    list_of_grids = []

    # Compute the nearest CT coordinate to each dose coordinate
    for index, (img_grid, dose_grid) in enumerate(grids):
        temp = list(set([utils.nearest(img_grid, val) for val in dose_grid]))
        temp.sort()
        list_of_grids.append(np.array(temp))

    # Need to offset by the amount coordinates were padded
    list_of_grids[2] -= padding

    try:
        try:
            dose_array = np.rollaxis(dose_dcm.pixel_array, 0, 3)
            interp = RegularGridInterpolator(list_of_grids, dose_array, method="linear")
        except ValueError:
            for index, (d_grid, i_grid) in enumerate(
                [(d_grid_x, i_grid_x), (d_grid_y, i_grid_y), (d_grid_z, i_grid_z)]
            ):
                if len(d_grid) > len(list_of_grids[index]):
                    # Calculate how many points overshoot
                    temp = [utils.nearest(i_grid, val) for val in d_grid]
                    xtra = [(x, n - 1)
                            for x, n in collections.Counter(temp).items() if n > 1]

                    if len(xtra) > 1:
                        lo_xtra, hi_xtra = (xtra[0][1], -xtra[1][1])
                        if index == 2:
                            cropped_dose_array = dose_array[:, :, lo_xtra:hi_xtra]
                        elif index == 1:
                            cropped_dose_array = dose_array[:, lo_xtra:hi_xtra]
                        else:
                            cropped_dose_array = dose_array[lo_xtra:hi_xtra]
                    else:
                        # Cropping for if the RTDOSE still extends
                        # This is likely unnecessary and may be removed later
                        if xtra[0][0] and index == 2:  # 511 and z
                            cropped_dose_array = dose_array[:, :, : -xtra[0][1]]
                        elif index == 2:  # 0 and z
                            cropped_dose_array = dose_array[:, :, xtra[0][1]:]
                        elif xtra[0][0] and index == 1:  # 511 and y
                            cropped_dose_array = dose_array[:, : -xtra[0][1]]
                        elif index == 1:  # 0 and y
                            cropped_dose_array = dose_array[:, xtra[0][1]:]
                        elif xtra[0][0]:  # 511 and x
                            cropped_dose_array = dose_array[: -xtra[0][1]]
                        else:  # 0 and x
                            cropped_dose_array = dose_array[: xtra[0][1]]
            interp = RegularGridInterpolator(list_of_grids, cropped_dose_array, method="linear")
            dose_array = cropped_dose_array
    except ValueError:
        if raises:
            raise ValueError(
                f"This patient, {patient_path}, has failed dose recontruction"
            )
        else:
            print(f"This patient, {patient_path}, has failed dose reconstruction")
    else:
        # Determine the upper and lower values for x, y, z for projection
        x_mm, y_mm, z_mm = [[t[0], t[-1]] for t in list_of_grids]
        total_pts = np.product([[y - x] for x, y in [x_mm, y_mm, z_mm]])
        grid = np.mgrid[x_mm[0]: x_mm[1], y_mm[0]: y_mm[1], z_mm[0]: z_mm[1]].reshape(3, total_pts)
        interp_pts = np.squeeze(np.array([grid]).T)
        interp_vals = interp(interp_pts)
        # This flipping may be redundant with flipping below
        interp_vol = interp_vals.reshape(x_mm[1] - x_mm[0],
                                         y_mm[1] - y_mm[0],
                                         z_mm[1] - z_mm[0])[..., ::-1]

        full_vol = np.zeros(img_dims)
        if z_1 < z_0:  # Because some images were flipped for interpolation
            interp_vol = interp_vol[..., ::-1]
            z_mm = [full_vol.shape[-1] - z_mm[1] - 1, full_vol.shape[-1] - z_mm[0] - 1]

        interp_lo, interp_hi = (0, interp_vol.shape[-1])

        if z_mm[1] > full_vol.shape[-1]:
            interp_hi = full_vol.shape[-1] - z_mm[1]
            z_mm[1] = z_mm[1] + interp_hi
        if z_mm[0] < 0:
            interp_lo = abs(z_mm[0])
            z_mm[0] = 0

        # Places the dose into the volume, which is usually correct
        full_vol[x_mm[0]: x_mm[1],
                 y_mm[0]: y_mm[1],
                 z_mm[0]: z_mm[1]] = interp_vol[..., interp_lo:interp_hi]

        # This is to ensure that d_max matches, if not, the image will be
        # adjusted to match accordingly
        slice_offset = utils.d_max_check(path=patient_path,
                                         volume=full_vol,
                                         printing=False)

        if np.any(slice_offset):
            full_vol = np.zeros(img_dims)
            o_x, o_y, o_z = slice_offset  # untested with x, y offset as none was needed
            x_mm, y_mm, z_mm = [[t[0], t[-1]] for t in list_of_grids]
            diff_lo, diff_hi = (0, interp_vol.shape[-1])

            # Offsets differently if coordinates are postive or negative
            if z_1 > z_0:
                # Crops, if necessary
                if (z_mm[1] + o_z) > full_vol.shape[-1]:
                    diff_hi = full_vol.shape[-1] - (o_z + z_mm[1])
                    z_mm[1] = z_mm[1] + diff_hi
                if (z_mm[0] + o_z) < 0:
                    diff_lo = abs(z_mm[0] + o_z)
                    z_mm[0] = z_mm[0] + diff_lo

                full_vol[x_mm[0] - o_x: x_mm[1] - o_x,
                         y_mm[0] - o_y: y_mm[1] - o_y,
                         z_mm[0] + o_z: z_mm[1] + o_z] = interp_vol[..., diff_lo:diff_hi]
            else:
                z_mm = [full_vol.shape[-1] - z_mm[1] - 1,
                        full_vol.shape[-1] - z_mm[0] - 1]
                # Crops, if necessary
                if (z_mm[1] - o_z) > full_vol.shape[-1]:
                    diff_hi = full_vol.shape[-1] - (z_mm[1] - o_z)
                    z_mm[1] = z_mm[1] + diff_hi
                if (z_mm[0] - o_z) < 0:
                    diff_lo = abs(z_mm[0] - o_z)
                    z_mm[0] = z_mm[0] + diff_lo

                full_vol[x_mm[0] - o_x: x_mm[1] - o_x,
                         y_mm[0] - o_y: y_mm[1] - o_y,
                         z_mm[0] - o_z: z_mm[1] - o_z] = interp_vol[..., diff_lo:diff_hi]

        # Adjusts for HFS if coordinates are negative
        if z_0 > z_1:
            full_vol = full_vol[..., ::-1]

        return full_vol * float(dose_dcm.DoseGridScaling)
