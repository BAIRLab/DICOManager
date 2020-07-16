#!/usr/bin/python3
import collections
import glob
import os
from pathlib import Path

import numpy as np
import pydicom
import skimage.draw as skdraw
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator


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
    if isinstance(key_list, dict):
        key_list = list(key_list.keys())

    def new_key(*args):
        return key_list.index(*args)

    return new_key


def _slice_thickness(dcm0, dcm1):
    # TODO: @evan-porter - Function is never called.
    
    """
    Function
    ----------
    Computes the slice thickness for a DICOM set
    -- *NOTE* Calculates based on slice location and instance number. Does not trust SliceThickness DICOM Header

    Parameters
    ----------
    dcm0, dcm1 : str or pydicom.dataset.FileDataset
        Either a string to the dicom path or a pydicom dataset

    Returns
    ----------
    slice_thickness : float
        A float representing the robustly calculated slice thickness
    """

    if not isinstance(dcm0, pydicom.dataset.FileDataset):
        dcm0 = pydicom.dcmread(dcm0)
    if not isinstance(dcm0, pydicom.dataset.FileDataset):
        dcm1 = pydicom.dcmread(dcm1)

    loc0 = dcm0.SliceLocation
    loc1 = dcm1.SliceLocation
    inst0 = dcm0.InstanceNumber
    inst1 = dcm1.InstanceNumber

    return abs((loc1 - loc0) / (inst1 - inst0))


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
        !! NOTE: number 3 is never used. Do we need it?
        4.Boolean indicating if image location / instances are flipped

    Notes
    ----------
    The values of high and low are for the highest and lowest instance,
        meaning high > low is not always true
    """
    # We need to save the location and instance, to know if counting up or down
    low = [float("inf"), 0]
    high = [-float("inf"), 0]

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
    n_slices = 1 + (abs(high[0] - low[0]) / thickness)

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
    # TODO: @evan-porter Support vs eliminate. Function is never called.
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

    if patient_path[0] == "~":
        patient_path = os.path.expanduser("~") + patient_path[1:]

    if patient_path[-1] != "/":
        patient_path += "/"

    dose_files = glob.glob(patient_path + "RTDOSE/*.dcm")
    ct_files = glob.glob(patient_path + "CT/*.dcm")
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

    d_grid_x = dose_iso[1] + dx * np.arange(dose_dims[0])
    d_grid_y = dose_iso[0] + dy * np.arange(dose_dims[1])
    d_grid_z = dose_iso[2] + dz * np.arange(dose_dims[2])

    i_grid_x = img_origin[1] + ix * np.arange(img_dims[0])
    i_grid_y = img_origin[0] + iy * np.arange(img_dims[1])
    i_grid_z = img_origin[2] + iz * np.arange(img_dims[2])

    if z_0 < z_1:
        i_grid_z = i_grid_z[::-1]

    volume_max = np.unravel_index(dose_volume.argmax(), dose_volume.shape)

    ct_max_mm = np.array(
        [i_grid_x[volume_max[0]], i_grid_y[volume_max[1]], i_grid_z[volume_max[2]]]
    )

    dose_coords = np.unravel_index(
        dose_dcm.pixel_array.argmax(), dose_dcm.pixel_array.shape
    )

    dose_max_mm = np.array(
        [d_grid_x[dose_coords[1]], d_grid_y[dose_coords[2]], d_grid_z[dose_coords[0]]]
    )

    if printing:
        with np.printoptions(
            formatter={"float": "{:>6.2f} ".format, "int": "{:>3d}    ".format}
        ):
            print(f"dose_volume max (voxels): {np.array(volume_max)}")
            print(f"D_max CT coordinate (mm): {ct_max_mm}")
            print(f"D_max RT coordinate (mm): {dose_max_mm}")
            print(f"D_max RT/CT coord abs(\u0394): {abs(ct_max_mm - dose_max_mm)}")
            print(f"1/2 CT voxel dimensions : {ct_voxel_size / 2}")

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

    modality_list = ["CT", "MR", "PET", "CBCT", "RTSTRUCT"]

    if path.is_dir():
        dir_contents = list(path.iterdir())
        if dir_contents[0].is_file():
            if dir_contents[0].suffix == ".dcm":
                seed_file = dir_contents[0]
            else:
                print(f"{dir_contents}")
                raise ValueError("Please provide a path to a DICOM File")
        elif dir_contents[0].is_dir():
            for d in dir_contents:
                if d.is_dir():
                    if d.name in modality_list:
                        print("Using first volume/RTSTRUCT found")
                        d_sub = [f for f in d.iterdir() if f.suffix == ".dcm"]
                        seed_file = d_sub[0]
                        print(f"{seed_file}")
                        break
                    else:
                        print("Using first dicom found")
                        d_globs = list(d.rglob("*.dcm"))
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
        if "ReferencedFrameOfReferenceSequence" in dir(dcmheader):
            target_frame_of_reference = dcmheader.ReferencedFrameOfReferenceSequence[
                0
            ].FrameOfReferenceUID
            pt_folders = list(path.parent.parent.iterdir())
            for p in pt_folders:
                if p.name in modality_list:
                    candidate_file = next(p.iterdir())
                    target_header = pydicom.dcmread(
                        str(candidate_file), stop_before_pixels=True
                    )
                    if "FrameOfReferenceUID" in dir(target_header):
                        if (
                            target_header.FrameOfReferenceUID
                            == target_frame_of_reference
                        ):
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
