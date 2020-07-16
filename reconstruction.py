#!/usr/bin/python3
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import numpy as np
import pydicom
import glob
import collections
import os
import cv2
from matplotlib import pyplot as plt

__author__ = ["Evan Porter", "David Solis", "Ron Levitin"]
__license__ = "Beaumont Artificial Intelligence Research Lab"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


def _nearest(array, value):
    """
    Function
    ----------
    Finds the nearest index of array to a given value

    Parameters
    ----------
    array : np.array
    value : float

    Returns
    ----------
    index : int
    """
    return np.abs(np.asarray(array) - value).argmin()


def _key_list_creator(key_list, *args):
    """
    Function
    ----------
    Smaller wrapper to create a key ordering list for the rtstruct func

    Parameters
    ----------
    key_list : list
        A list of the desired index order of structures, if present
    *args : list of arguments
        Arguements to be passed to key_list.index(*)

    Returns
    ----------
    Returns a function which yeilds the proper index of each structure
    """
    if type(key_list) is dict:
        key_list = list(key_list.keys())

    def new_key(*args):
        return key_list.index(*args)
    return new_key


def _slice_thickness(dcm0, dcm1):
    """
    Function
    ----------
    Computes the slice thickness for a DICOM set

    Parameters
    ----------
    dcm0, dcm1 : str or pydicom.dataset.FileDataset
        Either a string to the dicom path or a pydicom dataset

    Returns
    ----------
    slice_thickness : float
        A float representing the robustly calculated slice thickness
    
    Notes
    ----------
        Calculates based on slice location and instance number.
        Does not trust SliceThickness DICOM Header
    """
    if type(dcm0) != pydicom.dataset.FileDataset:
        dcm0 = pydicom.dcmread(dcm0)
    if type(dcm1) != pydicom.dataset.FileDataset:
        dcm1 = pydicom.dcmread(dcm1)

    loc0 = dcm0.SliceLocation
    loc1 = dcm1.SliceLocation
    inst0 = dcm0.InstanceNumber
    inst1 = dcm1.InstanceNumber

    return abs((loc1-loc0) / (inst1-inst0))


def _img_dims(dicom_list):
    """
    Function
    ----------
    Computation of the image dimensions for slice thickness and number
        of z slices in total

    Parameters
    ----------
    dicom_list : list
        A list of the paths to every dicom for the given image

    Returns
    ----------
    (thickness, n_slices, low, high, flip) : (float, int, float, float, boolean)
        0.Slice thickness computed from dicom locations, not header
        1.Number of slices, computed from dicom locations, not header
        2.Patient coordinate system location of lowest instance
        3.Patient coordinate system location of highest instance
        4.Boolean indicating if image location / instances are flipped

    Notes
    ----------
    The values of high and low are for the highest and lowest instance,
        meaning high > low is not always true
    """
    # We need to save the location and instance, to know if counting up or down
    low = [float('inf'), 0]
    high = [-float('inf'), 0]

    instances = []
    for f in dicom_list:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if float(ds.SliceLocation) < low[0]:
            low = [float(ds.SliceLocation), round(ds.InstanceNumber)]
        if float(ds.SliceLocation) > high[0]:
            high = [float(ds.SliceLocation), round(ds.InstanceNumber)]
        instances.append(round(ds.InstanceNumber))
    instances.sort()

    thickness = _slice_thickness(dicom_list[0], dicom_list[1])
    n_slices = 1 + (abs(high[0]-low[0]) / thickness)

    # We need to cover for if instance 1 is missing
    # Unfortunately, we don't know how many upper instances could be missing
    if 1 != min(instances):
        diff = min(instances) - 1
        n_slices += diff
        if low[1] < high[1]:
            low[0] -= thickness * diff
        else:
            high[0] += thickness * diff

    flip = True
    if low[1] > high[1]:
        flip = False
        low[0], high[0] = high[0], low[0]

    return thickness, round(n_slices), low[0], high[0], flip


def _d_max_coords(patient_path, dose_volume, printing=True):
    """
    Function
    ----------
    Used to determine the coordinates of D_max, as a dose reconstruction
        sanity check. Can be compared against the TPS or MIM

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    dose_volume : array
        An array of the dose volume, as returned form the dose
            reconstruction function
    printing : bool (Default = True)
        Prints a formatting of the results, along with the voxel size

    Returns
    ----------
    (volume_max, ct_max_mm, dose_max_mm, ct_voxel_size) : (float, float, float, float)
        0.The dose maximum coordinates (x, y, z) in voxels of the volume
        1.The dose maximum coordinates (x, y, z) in mm relative to isocenter
            in the CT coordinate system
        2.The dose maximum coordinates (x, y, z) in mm relative to isocenter
            in the RTDOSE coordinate system
        3.The CT voxel size for (x, y, z) in mm

    Notes
    ----------
    Due to the RTDOSE computation, it is likely in a slighly different
        coordinate system than the CT coordinates. But, the slice difference
        should be < 1/2 * voxel size of the CT coordinates
    """
    if patient_path[0] == '~':
        patient_path = os.path.expanduser('~') + patient_path[1:]

    if patient_path[-1] != '/':
        patient_path += '/'

    dose_files = glob.glob(patient_path + 'RTDOSE/*.dcm')
    ct_files = glob.glob(patient_path + 'CT/*.dcm')
    ct_files.sort()

    dose_dcm = pydicom.dcmread(dose_files[0])
    for ct_file in ct_files:
        ct_dcm = pydicom.dcmread(ct_file, stop_before_pixels=True)
        if ct_dcm.InstanceNumber == 1:
            z_0 = float(ct_dcm.ImagePositionPatient[-1])
        if ct_dcm.InstanceNumber == len(ct_files):
            z_1 = float(ct_dcm.ImagePositionPatient[-1])

    img_dims = (ct_dcm.Rows, ct_dcm.Columns, len(ct_files))
    img_origin = np.array([*ct_dcm.ImagePositionPatient[:2], min(z_0, z_1)])
    ix, iy, iz = (*ct_dcm.PixelSpacing, ct_dcm.SliceThickness)
    ct_voxel_size = np.array([ix, iy, iz])

    dose_dims = np.rollaxis(dose_dcm.pixel_array, 0, 3).shape
    dose_iso = np.array(dose_dcm.ImagePositionPatient)
    dx, dy, dz = (*dose_dcm.PixelSpacing, dose_dcm.SliceThickness)

    d_grid_x = dose_iso[1]+dx * np.arange(dose_dims[0])
    d_grid_y = dose_iso[0]+dy * np.arange(dose_dims[1])
    d_grid_z = dose_iso[2]+dz * np.arange(dose_dims[2])

    i_grid_x = img_origin[1]+ix * np.arange(img_dims[0])
    i_grid_y = img_origin[0]+iy * np.arange(img_dims[1])
    i_grid_z = img_origin[2]+iz * np.arange(img_dims[2])

    if z_0 < z_1:
        i_grid_z = i_grid_z[::-1]

    volume_max = np.unravel_index(dose_volume.argmax(),
                                  dose_volume.shape)

    ct_max_mm = np.array([i_grid_x[volume_max[0]],
                          i_grid_y[volume_max[1]],
                          i_grid_z[volume_max[2]]])

    dose_coords = np.unravel_index(dose_dcm.pixel_array.argmax(),
                                   dose_dcm.pixel_array.shape)

    dose_max_mm = np.array([d_grid_x[dose_coords[1]],
                            d_grid_y[dose_coords[2]],
                            d_grid_z[dose_coords[0]]])

    if printing:
        with np.printoptions(formatter={'float': '{:>6.2f} '.format,
                                        'int': '{:>3d}    '.format}):
            print(f'dose_volume max (voxels): {np.array(volume_max)}')
            print(f'D_max CT coordinate (mm): {ct_max_mm}')
            print(f'D_max RT coordinate (mm): {dose_max_mm}')
            print(
                f'D_max RT/CT coord abs(\u0394): {abs(ct_max_mm - dose_max_mm)}')
            print(f'1/2 CT voxel dimensions : {ct_voxel_size / 2}')

    return (volume_max, ct_max_mm, dose_max_mm, ct_voxel_size)


def _d_max_check(path, volume, printing):
    """
    Function
    ----------
    Calculates the necessary offset for aligning the Dose Max value

    Parameters
    ----------
    patient_path : str
        A path directing towards a patient database in the following format:
            MRN/[CT,PET,RTSTRUCT,RTDOSE]/*.dcm
    dose_volume : array
        An array of the dose volume, as returned form the dose
            reconstruction function
    printing : bool (Default = True)
        Prints a formatting of the results, along with the voxel size

    Returns
    ----------
    offset : np.array
        An array of the voxel offset in the [x, y, z] axis
    """
    _, ct_max, dose_max, v_size = _d_max_coords(path, volume, printing)
    offset = [0, 0, 0]

    for i, diff in enumerate(ct_max - dose_max):
        pad = 30  # Chosen because > 30 offset likely indicates greater issues
        low, high, spacing = (-pad * v_size[i], pad * v_size[i], v_size[i])
        offset_list = np.arange(low, high, spacing)
        offset[i] = _nearest(offset_list, diff) - pad

    return np.array(offset, dtype=int)


def _find_series_slices(path, find_associated=False):
    """
    Function
    ----------
    Given the path to a slice of a volume, checks all files within the
        directory for the remaining slices of a given volume using
        the SeriesInstanceUID

    Parameters
    ----------
    path : pathlib.Path
        A path to a DICOM image volume slice
    find_associated : bool (Default = False)
        A boolean designating finding a volume associated to a
            associated RTSTRUCT based on SeriesInstanceUID

    Returns
    ----------
    volume_slice_list : list
        A list of files belonging to an image volume
    """
    if not isinstance(path, Path):
        path = Path(path)

    modality_list = ['CT', 'MR', 'PET', 'CBCT', 'RTSTRUCT']

    if path.is_dir():
        dir_contents = list(path.iterdir())
        if dir_contents[0].is_file():
            if dir_contents[0].suffix == '.dcm':
                seed_file = dir_contents[0]
            else:
                print(f"{dir_contents}")
                raise ValueError("Please provide a path to a DICOM File")
        elif dir_contents[0].is_dir():
            for d in dir_contents:
                if d.is_dir():
                    if d.name in modality_list:
                        print("Using first volume/RTSTRUCT found")
                        d_sub = [f for f in d.iterdir() if f.suffix == '.dcm']
                        seed_file = d_sub[0]
                        print(f"{seed_file}")
                        break
                    else:
                        print("Using first dicom found")
                        d_globs = list(d.rglob('*.dcm'))
                        seed_file = d_globs[0]
                        print(f"{seed_file}")
                        break
                elif d.is_file() and d.suffix == ".dcm":
                    seed_file = d
                    print(f"{seed_file}")
                    break
        else:
            print("Check path")
            return None
    elif path.is_file():
        seed_file = path
    else:
        raise ValueError("Path is not a file or a directory")

    if find_associated:
        dcmheader = pydicom.dcmread(str(seed_file), stop_before_pixels=True)
        if 'ReferencedFrameOfReferenceSequence' in dir(dcmheader):
            target_frame_of_reference = dcmheader.ReferencedFrameOfReferenceSequence[
                0].FrameOfReferenceUID
            pt_folders = list(path.parent.parent.iterdir())
            for p in pt_folders:
                if p.name in modality_list:
                    candidate_file = next(p.iterdir())
                    target_header = pydicom.dcmread(
                        str(candidate_file), stop_before_pixels=True)
                    if 'FrameOfReferenceUID' in dir(target_header):
                        if target_header.FrameOfReferenceUID == target_frame_of_reference:
                            seed_file = candidate_file
                            break
                        else:
                            pass
        else:
            print("No Frame of Reference Found")
            return None

    file_list = list(seed_file.parent.iterdir())

    volume_slice_list = []
    series = []
    for f in file_list:
        dcmheader = pydicom.dcmread(str(f), stop_before_pixels=True)
        if not series:
            series = dcmheader.SeriesInstanceUID
            volume_slice_list.append(str(f))
        elif dcmheader.SeriesInstanceUID == series:
            volume_slice_list.append(str(f))

    volume_slice_list.sort()

    return volume_slice_list


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

    # If path is a string, conver to pathlib.Path
    if not isinstance(patient_path, Path):
        patient_path = Path(patient_path).expanduser()

    # If directly given a dicom
    if patient_path.is_file():
        volume_slices = _find_series_slices(patient_path)
        dcmheader = pydicom.dcmread(volume_slices[0])
    elif patient_path.is_dir():
        patient_path = patient_path / 'MR'
        if patient_path.is_dir():
            volume_slices = _find_series_slices(patient_path)
            dcmheader = pydicom.dcmread(volume_slices[0])
        else:
            err_msg = f"No MR folder in patient path: {patient_path.parent}"
            if raises:
                raise ValueError(err_msg)
            else:
                print(err_msg)

    slice_thick, n_z, loc0, loc1, flip = _img_dims(volume_slices)
    image_array = np.zeros((*dcmheader.pixel_array.shape,
                            n_z), dtype='float32')

    try:
        for slice_file in volume_slices:
            ds = pydicom.dcmread(str(slice_file))
            z_loc = round(abs((loc0-ds.SliceLocation) / slice_thick))
            image_array[:, :, z_loc] = ds.pixel_array
    except IndexError:
        if raises:
            raise IndexError(f'There is a discontinuity in {patient_path}/MR')
        else:
            print(f'This is a discontinuity in {patient_path}/MR')
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
            in the same format as the recieved list of names, or order of keys
        If list, then only items in the list will be output. If a dict, then
            each list will map to the key and saved the respective index
    raises : bool (Default = False)
        Determines if errors are raised or not. As false, only printing occurs

    Raises
    ----------
    ValueEror
        Raised if the RTSTRUCT and CT are not registered
    ValueError
        Raised if values are missing from an RTSTRUCT contour sequence

    Returns
    ----------
    rtstruct_array : np.array
        A reconstructed RTSTRUCT array in the shape of [struct, x, y, z], where
            x, y, z are the same dimensions as the registered CT volume
    """
    # TODO: Add way to get all contours in the RTSTRUCT
    # Check path type
    # -- if dicom file, check directory for related dicoms
    # -- if directory, check for "[CT, MRI, CBCT, PET, RTSTRUCT, RTDOSE]" folder and read those in.

    # If path is a string, convert to pathlib.Path
    if not isinstance(patient_path, Path):
        patient_path = Path(patient_path).expanduser()

    # find the RTSTRUCT dcm file
    if patient_path.is_file() and patient_path.suffix == ".dcm":
        struct_file = patient_path
    elif patient_path.is_dir():
        patient_path_sub = patient_path / 'RTSTRUCT'
        if patient_path_sub.is_dir():
            struct_file = next(patient_path_sub.iterdir())
        else:
            err_msg = f"No RTSTRUCT folder in patient path: {patient_path.parent}"
            if raises:
                raise ValueError(err_msg)
            else:
                print(err_msg)

    # Open RTSTRUCT Header
    struct_dcm = pydicom.dcmread(str(struct_file))
    if not struct_dcm.Modality == 'RTSTRUCT':
        err_msg = f"DICOM header for file {struct_file} is of Modality {struct_dcm.Modality}"
        raise ValueError(err_msg) if raises else print(err_msg)

    # Find the associated volume dicoms and read them.
    # Need the associated volume for the image space information
    volume_slice_files = _find_series_slices(
        str(struct_file), find_associated=True)

    for vfile in volume_slice_files:
        volume_dcm = pydicom.dcmread(vfile, stop_before_pixels=True)
        if volume_dcm.InstanceNumber == 1:
            break

    _, vol_n_z, _, _, _ = _img_dims(volume_slice_files)
    dimensions = (volume_dcm.Rows, volume_dcm.Columns, vol_n_z)
    img_origin = np.array(volume_dcm.ImagePositionPatient)
    ix, iy, iz = (*volume_dcm.PixelSpacing, volume_dcm.SliceThickness)

    # inner function to convert the points to coords
    def _points_to_coords(contour_data, img_origin, ix, iy, iz):
        points = np.array(
            np.round(abs(contour_data - img_origin) / [ix, iy, abs(iz)]), dtype=int)
        return points

    # This function requires a list of the contours being looked for. Can be dict or list
    if type(wanted_contours) is dict:
        wanted_contours = dict((k.lower(), v)
                               for k, v in wanted_contours.items())
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
            all_points = np.empty((0, 3))

            for contour_slice in contour_list:
                try:
                    contour_data = np.array(
                        contour_slice.ContourData).reshape(-1, 3)
                except ValueError:
                    err_msg = f'Contour {contour.ROIName} in {struct_file} is corrupt'
                    raise ValueError(err_msg) if raises else print(err_msg)

                points = _points_to_coords(
                    contour_data, img_origin, ix, iy, iz)
                coords = np.array([points[:, :2]], dtype=np.int32)

                # scimage.draw.Polygon is incorrect, use cv2.fillPoly instead
                poly_2D = np.zeros(dimensions[:2])
                cv2.fillPoly(poly_2D, coords, 1)
                fill_array[:, :, points[0, 2]] += poly_2D

        # Protect against any overlaps in the contour
        fill_array[fill_array > 2] = 1
        masks.append(fill_array)
    # Reorders the list to match the wanted contours
    key_list = _key_list_creator(wanted_contours)
    ordered = [masks[contours.index(x)]
               for x in sorted(contours, key=key_list)]
    return np.array(ordered, dtype='bool')


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
    if patient_path[0] == '~':
        patient_path = os.path.expanduser('~') + patient_path[1:]

    if patient_path[-1] != '/':
        patient_path += '/'

    ct_files = glob.glob(patient_path + 'CT/*.dcm')
    ct_files.sort()
    ct_dcm = pydicom.dcmread(ct_files[0])

    ct_thick, ct_n_z, ct_loc0, ct_loc1, flip = _img_dims(ct_files)
    ct_array = np.zeros((*ct_dcm.pixel_array.shape,
                         ct_n_z), dtype='float32')

    try:
        for ct_file in ct_files:
            ds = pydicom.dcmread(ct_file)
            z_loc = round(abs((ct_loc0-ds.SliceLocation) / ct_thick))
            ct_array[:, :, z_loc] = ds.pixel_array
    except IndexError:
        if raises:
            raise IndexError(f'There is a discontinuity in {patient_path}/CT')
        else:
            print(f'This is a discontinuity in {patient_path}/CT')
    else:
        if flip:
            ct_array = ct_array[..., ::-1]
        if HU:
            return ct_array * ct_dcm.RescaleSlope+ct_dcm.RescaleIntercept
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
    ValueEror
        Raised if the PET and CT are not registered
    IndexError
        Raised if any instances are missing from the PET directory

    Returns
    ----------
    pet_array : np.array
        A reconstructed PET image in the same dimensions as the registered CT
    """
    if patient_path[0] == '~':
        patient_path = os.path.expanduser('~') + patient_path[1:]

    if patient_path[-1] != '/':
        patient_path += '/'

    if os.path.isdir(patient_path + 'CT' + str(path_mod)):
        ct_files = glob.glob(patient_path + 'CT' + str(path_mod) + '/*.dcm')
    else:
        ct_files = glob.glob(patient_path + 'CT/*.dcm')

    ct_dcm = pydicom.dcmread(ct_files[0], stop_before_pixels=True)

    pet_files = glob.glob(patient_path + 'PET' + str(path_mod) + '/*.dcm')
    pet_dcm = pydicom.dcmread(pet_files[0])

    pet_thick, pet_n_z, pet_loc0, pet_loc1, flip = _img_dims(pet_files)
    pet_array = np.zeros((*pet_dcm.pixel_array.shape,
                          pet_n_z), dtype='float32')

    if pet_dcm.FrameOfReferenceUID != ct_dcm.FrameOfReferenceUID:
        if raises:
            raise ValueError(
                f'These two images are not registered: {ct_files[0]} & {pet_files[0]}')
        else:
            print(
                f'These two images are not registered: {ct_files[0]} & {pet_files[0]}')
    else:
        try:
            for index, pet_file in enumerate(pet_files):
                ds = pydicom.dcmread(pet_file)
                z_loc = round(abs((pet_loc0 - ds.SliceLocation) / pet_thick))
                pet_array[:, :, z_loc] = ds.pixel_array
        except IndexError:
            if raises:
                raise IndexError('There is a discontinuity in your images')
            else:
                print(f'There is a discontinuity in {patient_path}PET')
        else:
            if flip:
                pet_array = pet_array[..., ::-1]
            rescaled = pet_array * pet_dcm.RescaleSlope + pet_dcm.RescaleIntercept
            patient_weight = 1000 * float(pet_dcm.PatientWeight)
            total_dose = float(
                pet_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
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
    if patient_path[0] == '~':
        patient_path = os.path.expanduser('~') + patient_path[1:]

    if patient_path[-1] != '/':
        patient_path += '/'

    dose_files = glob.glob(patient_path + 'RTDOSE/*.dcm')
    ct_files = glob.glob(patient_path + 'CT/*.dcm')
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
        print('This patient has multiple CT image sets or missing files')
        return 0

    img_dims = (ct_dcm.Rows, ct_dcm.Columns, len(ct_files))
    ix, iy, iz = (*ct_dcm.PixelSpacing, ct_dcm.SliceThickness)

    dose_iso = np.array(dose_dcm.ImagePositionPatient)
    dose_dims = np.rollaxis(dose_dcm.pixel_array, 0, 3).shape
    dx, dy, dz = (*dose_dcm.PixelSpacing, dose_dcm.SliceThickness)

    d_grid_x = dose_iso[1]+dx * np.arange(dose_dims[0])
    d_grid_y = dose_iso[0]+dy * np.arange(dose_dims[1])
    d_grid_z = dose_iso[2]+dz * np.arange(dose_dims[2])

    padding = 25  # Padding since some RTDOSE extend outside the CT volume
    i_grid_x = img_origin[1]+ix * np.arange(img_dims[0])
    i_grid_y = img_origin[0]+iy * np.arange(img_dims[1])
    i_grid_z = img_origin[2]+iz * np.arange(-padding, img_dims[2] + padding)
    i_grid_z = i_grid_z[::-1]  # Likely redundant with lower flips

    grids = [(i_grid_x, d_grid_x), (i_grid_y, d_grid_y), (i_grid_z, d_grid_z)]
    list_of_grids = []

    # Compute the nearest CT coordinate to each dose coordinate
    for index, (img_grid, dose_grid) in enumerate(grids):
        temp = list(set([_nearest(img_grid, val) for val in dose_grid]))
        temp.sort()
        list_of_grids.append(np.array(temp))

    # Need to offset by the amount coordinates were padded
    list_of_grids[2] -= padding

    try:
        try:
            dose_array = np.rollaxis(dose_dcm.pixel_array, 0, 3)
            interp = RegularGridInterpolator(
                list_of_grids, dose_array, method='linear')
        except ValueError:
            for index, (d_grid, i_grid) in enumerate([(d_grid_x, i_grid_x),
                                                      (d_grid_y, i_grid_y),
                                                      (d_grid_z, i_grid_z)]):
                if len(d_grid) > len(list_of_grids[index]):
                    # Calculate how many points overshoot
                    temp = [_nearest(i_grid, val) for val in d_grid]
                    xtra = [(x, n - 1)
                            for x, n in collections.Counter(temp).items() if n > 1]
                    if len(xtra) > 1:
                        lo_xtra, hi_xtra = (xtra[0][1], -xtra[1][1])
                        if index == 2:
                            cropped_dose_array = dose_array[:,
                                                            :,
                                                            lo_xtra: hi_xtra]
                        elif index == 1:
                            cropped_dose_array = dose_array[:,
                                                            lo_xtra: hi_xtra]
                        else:
                            cropped_dose_array = dose_array[lo_xtra: hi_xtra]
                    else:
                        # Cropping for if the RTDOSE still extends
                        # This is likely unnecessary and may be removed later
                        if xtra[0][0] and index == 2:  # 511 and z
                            cropped_dose_array = dose_array[:, :, :-xtra[0][1]]
                        elif index == 2:  # 0 and z
                            cropped_dose_array = dose_array[:, :, xtra[0][1]:]
                        elif xtra[0][0] and index == 1:  # 511 and y
                            cropped_dose_array = dose_array[:, :-xtra[0][1]]
                        elif index == 1:  # 0 and y
                            cropped_dose_array = dose_array[:, xtra[0][1]:]
                        elif xtra[0][0]:  # 511 and x
                            cropped_dose_array = dose_array[:-xtra[0][1]]
                        else:  # 0 and x
                            cropped_dose_array = dose_array[:xtra[0][1]]
            interp = RegularGridInterpolator(
                list_of_grids, cropped_dose_array, method='linear')
            dose_array = cropped_dose_array
    except ValueError:
        if raises:
            raise ValueError(
                f'This patient, {patient_path}, has failed dose recontruction')
        else:
            print(
                f'This patient, {patient_path}, has failed dose reconstruciton')
    else:
        # Determine the upper and lower values for x, y, z for projection
        x_mm, y_mm, z_mm = [[t[0], t[-1]] for t in list_of_grids]
        total_pts = np.product([[y - x] for x, y in [x_mm, y_mm, z_mm]])
        interp_pts = np.squeeze(np.array([np.mgrid[x_mm[0]:x_mm[1],
                                                   y_mm[0]:y_mm[1],
                                                   z_mm[0]:z_mm[1]].reshape(3, total_pts)]).T)
        interp_vals = interp(interp_pts)
        # This flipping may be redundant with flipping below
        interp_vol = interp_vals.reshape(x_mm[1]-x_mm[0],
                                         y_mm[1]-y_mm[0],
                                         z_mm[1]-z_mm[0])[..., ::-1]

        full_vol = np.zeros(img_dims)
        if z_1 < z_0:  # Because some images were flipped for interpolation
            interp_vol = interp_vol[..., ::-1]
            z_mm = [full_vol.shape[-1]-z_mm[1]-1,
                    full_vol.shape[-1]-z_mm[0]-1]

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
                 z_mm[0]: z_mm[1]] = interp_vol[..., interp_lo: interp_hi]

        # This is to ensure that d_max matches, if not, the image will be
        # adjusted to match accordingly
        slice_offset = _d_max_check(path=patient_path,
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
                if (z_mm[1]+o_z) > full_vol.shape[-1]:
                    diff_hi = full_vol.shape[-1] - (o_z+z_mm[1])
                    z_mm[1] = z_mm[1] + diff_hi
                if (z_mm[0]+o_z) < 0:
                    diff_lo = abs(z_mm[0]+o_z)
                    z_mm[0] = z_mm[0] + diff_lo

                full_vol[x_mm[0]-o_x: x_mm[1]-o_x,
                         y_mm[0]-o_y: y_mm[1]-o_y,
                         z_mm[0]+o_z: z_mm[1]+o_z] = interp_vol[..., diff_lo: diff_hi]
            else:
                z_mm = [full_vol.shape[-1]-z_mm[1]-1,
                        full_vol.shape[-1]-z_mm[0]-1]
                # Crops, if necessary
                if (z_mm[1]-o_z) > full_vol.shape[-1]:
                    diff_hi = full_vol.shape[-1] - (z_mm[1]-o_z)
                    z_mm[1] = z_mm[1] + diff_hi
                if (z_mm[0]-o_z) < 0:
                    diff_lo = abs(z_mm[0]-o_z)
                    z_mm[0] = z_mm[0] + diff_lo

                full_vol[x_mm[0]-o_x: x_mm[1]-o_x,
                         y_mm[0]-o_y: y_mm[1]-o_y,
                         z_mm[0]-o_z: z_mm[1]-o_z] = interp_vol[..., diff_lo: diff_hi]

        # Adjusts for HFS if coordinates are negative
        if z_0 > z_1:
            full_vol = full_vol[..., ::-1]

        return full_vol * float(dose_dcm.DoseGridScaling)


# Code below is for viewing volumes


def show_volume(volume_array, rows=2, cols=6, figsize=None):
    """
    Function
    ----------
    Displays the given image volume

    Paramters
    ----------
    volume_array : np.array
        Numpy arrays designating image and contour volumes
    rows, cols : int (Default: rows = 2, cols = 6)
        Intergers representing the rows, columns of the displayed images
    figsize : int (Default = None)
        An integer representing the pyplot figure size
    """
    if not figsize:
        figsize = (cols * 4, rows * 3)

    plt.figure(figsize=figsize)
    zmax = volume_array.shape[-1]
    imgs_out = rows * cols
    skip = zmax // imgs_out
    extra = zmax % imgs_out
    for i in range(imgs_out):
        plt.subplot(rows, cols, i+1)
        z = i * skip + int(extra/2 + skip/2)
        plt.imshow(volume_array[:, :, z], cmap='gray', interpolation='none')
        plt.axis('off')
        plt.title(f"z={z}")
    plt.show


def show_contours(volume_arr, contour_arr, rows=2, cols=6,
                  orientation='axial', aspect=3, figsize=None):
    """
    Function
    ----------
    Displays a volume with overlaid contours

    Paramters
    ----------
    volume_arr, contour_arr : np.array
        Numpy arrays designating image and contour volumes
    rows, cols : int (Default: rows = 2, cols = 2)
        Intergers representing the rows, columns of the displayed images
    aspect : float (Default = 3)
        An aspect ratio for image display
    orientation : str (Default = 'axial')
        Designates the axis from which to display the images. Options include
            'axial', 'sagittal' and 'coronal'
    figsize : int (Default = None)
        An integer representing the pyplot figure size
    """

    # Currently hard coded the aspect ratio for MRI context. May need to adapt it to make it more adaptible.

    if not figsize:
        figsize = (cols * 4, rows * 3)

    masked_contour_arr = np.ma.masked_equal(contour_arr, 0)

    plt.figure(figsize=figsize)
    xmax, ymax, zmax = volume_arr.shape
    imgs_out = rows * cols

    if orientation == "axial":
        max_dim = zmax
        dim = "z"
    elif orientation == "coronal":
        max_dim = xmax
        dim = "x"
    elif orientation == "sagittal":
        max_dim = ymax
        dim = "y"
    else:
        raise NameError('The orientation provided is not an option')

    skip = max_dim // imgs_out
    extra = max_dim % imgs_out
    slice_index = list(range(int(skip/2 + extra/2),
                             max_dim - int(skip/2 + extra/2), skip))

    for i, s in enumerate(slice_index):
        plt.subplot(rows, cols, i+1)

        if orientation == "axial":
            plt.imshow(volume_arr[:, :, s], cmap='gray', interpolation='none')
            plt.imshow(masked_contour_arr[:, :, s],
                       cmap='Pastel1', interpolation='none', alpha=0.5)
        elif orientation == "coronal":
            plt.imshow(np.rollaxis(volume_arr[s, :, ::-1], 1, 0), cmap='gray',
                       interpolation='none', aspect=aspect)
            plt.imshow(np.rollaxis(masked_contour_arr[s, :, ::-1], 1, 0),
                       cmap='Pastel1', interpolation='none', alpha=0.5, aspect=aspect)
        elif orientation == "sagittal":
            plt.imshow(np.rollaxis(volume_arr[:, s, ::-1], 1, 0), cmap='gray',
                       interpolation='none', aspect=aspect)
            plt.imshow(np.rollaxis(masked_contour_arr[:, s, ::-1], 1, 0),
                       cmap='Pastel1', interpolation='none', alpha=0.5, aspect=aspect)

        plt.axis('off')
        plt.title(f"{dim}={s}")

    plt.show()
