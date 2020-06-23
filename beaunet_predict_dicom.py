#! /usr/bin/python
#
# This script will use a trained Beaunet model to run on a directory of DICOM-format image files (CT only) and
# write a DICOM RTSTRUCT file that contains predicted contours.
#


# May want to do less specific imports ... 
import argparse, h5py, numpy as np, os, random, time
from functools import partial
from itertools import repeat
import concurrent.futures
from pydicom import dataset, read_file, sequence, uid

# Will need to identify how Shapely is used, it is quite slow...
from shapely import speedups
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

from skimage.transform import resize
from skimage.measure import label, regionprops

# Make shapely go fast
if speedups.available:
    speedups.enable()

# For randomness
random.seed(datetime.now())
np.random.seed(datetime.now())

# Config parser
parser = argparse.ArgumentParser(description="Run BEAUNET on a directory of CT images in DICOM format and write an RTSTRUCT with predicted contours")
parser.add_argument("--input", required=True, help="The directory containing DICOM files to use as input")
parser.add_argument('--stats', required=True, help="The directory containing the stats files needed for modeling")
parser.add_argument("--output", required=True, help="The output file to write")
parser.add_argument("--hierarchical", required=False, action='store_true', help='Run in hierachical mode')
parser.add_argument("--postprocess", required=False, action='store_true', help='Postprocess contours')
args = parser.parse_args()

INPUT_PATH = args.input
STATS_PATH = args.stats
OUTPUT_FILENAME = args.output
HIERARCHICAL_MODE = args.hierarchical
POSTPROCESS = args.postprocess

if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print('BEAUNET AUTO-CONTOURING SYSTEM')
print('Opening \'{0:s}\' in {1:s} mode'.format(INPUT_PATH, 'DIRECT' if not HIERARCHICAL_MODE else 'HIERARCHICAL'))
print('Output to \'{0:s}\''.format(OUTPUT_FILENAME))

if POSTPROCESS:
    print('POSTPROCESSING is ENABLED')
else:
    print('POSTPROCESSING is DISABLED')

def __prepare_coordinate_mapping(image_slice):
    """
    Given a DICOM CT image slice, returns a numpy array with the coordinates of each pixel.
    :param image_slice: A pydicom dataset representing a CT slice in DICOM format.
    :return: A numpy array of shape Mx2 where M is image_slice.rows x image_slice.cols, the number of(x,y) pairs
             representing coordinates of each pixel.
    """

    M = np.array(
        [[np.array(image_slice.ImageOrientationPatient[0:3])[0] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[0] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[0]],

         [np.array(image_slice.ImageOrientationPatient[0:3])[1] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[1] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[1]],

         [np.array(image_slice.ImageOrientationPatient[0:3])[2] * image_slice.PixelSpacing[1],
          np.array(image_slice.ImageOrientationPatient[3:6])[2] * image_slice.PixelSpacing[0],
          0,
          np.array(image_slice.ImagePositionPatient)[2]],

         [0,
          0,
          0,
          1]
         ])

    pixel_coord_array = np.zeros((image_slice.Rows, image_slice.Columns), dtype=(float, 3))
    pixel_idx_array = np.indices((image_slice.Rows, image_slice.Columns))

    it = np.nditer(op=[pixel_idx_array[0],  # Array of pixel row indices (j)
                       pixel_idx_array[1],  # Array of pixel col indices (i)
                       pixel_coord_array[:, :, 0],  # Output array of pixel x coords
                       pixel_coord_array[:, :, 1],  # Output array of pixel y coords
                       pixel_coord_array[:, :, 2]],  # Output array of pixel z coords

                   flags=['external_loop', 'buffered'],
                   op_flags=[['readonly'],
                             ['readonly'],
                             ['writeonly', 'no_broadcast'],
                             ['writeonly', 'no_broadcast'],
                             ['writeonly', 'no_broadcast']])

    for (j, i, Px, Py, Pz) in it:
        C = np.array([i, j, 0, 1])
        P = np.dot(M, C)

        Px[...] = P[0]
        Py[...] = P[1]
        Pz[...] = P[2]

    return pixel_coord_array


def __threaded_geom_op(item):
    try:
        image_ds = item[0]
        y_elem = item[1]
        class_list = item[2]

        y_elem = resize(y_elem, output_shape=(image_ds.Rows, image_ds.Columns), order=0, mode='constant', preserve_range=True)

        # Process prediction into contours
        contour_list_pred = {}  # This will contain contours for this slice represented as lists of polygons
        p_s_x = float(image_ds.PixelSpacing[0])
        p_s_y = float(image_ds.PixelSpacing[1])
        p_coords = __prepare_coordinate_mapping(
            image_ds)  # X,Y,3 matrix containing at each voxel the coordinates of the center of that voxel

        for roi_name, label_img in zip(class_list[1:], np.rollaxis(y_elem, 2, 0)[1:, ...]):
            connected_comps_pred = label(label_img, connectivity=2)
            contour_list_roi = []
            region_count = 0
            
            for region in regionprops(connected_comps_pred):
                region_count += 1
                multi_poly_list = []

                for point in MultiPoint([(p_coords[p[0], p[1], :]) for p in region.coords]):
                    # We have converted now from pixel coordinates to RCS coordinates. When
                    # we divide pixelspacing by 2, we get all sorts of ugly disjoint contours.
                    # Instead, we divide by 1.5 for a bit of overlap.
                    point_poly = Polygon([(point.x - p_s_x / 1.5, point.y - p_s_y / 1.5, point.z),
                                          (point.x + p_s_x / 1.5, point.y - p_s_y / 1.5, point.z),
                                          (point.x + p_s_x / 1.5, point.y + p_s_y / 1.5, point.z),
                                          (point.x - p_s_x / 1.5, point.y + p_s_y / 1.5, point.z)])

                    multi_poly_list.append(point_poly)

                dissolve_poly = unary_union(MultiPolygon(multi_poly_list))
                shape_polys = []

                if isinstance(dissolve_poly, Polygon):
                    shape_polys.append(dissolve_poly)
                elif isinstance(dissolve_poly, MultiPolygon):
                    shape_polys.extend([poly for poly in dissolve_poly])

                # Old and busted: This region's shapes get coordified at the time of region processing.
                # for poly in shape_polys:
                #     coord_list = poly.exterior.coords[:]
                #     coord_list = [(x[0], x[1], x[2]) for x in coord_list]
                #     contour_list_roi.append(coord_list)

                # New hotness: We store the polygons for the region and postprocess the whole ROI at once.

                contour_list_roi.extend(shape_polys)

            # If there are any contours for this ROI, we'll add it to the list of contours for this slice
            # Postprocessing based on contour geometry, etc, may go here if we store it somewhere.

            if contour_list_roi:
                postprocessed_roi_list = contour_list_roi

                if POSTPROCESS:
                    # Step 1: Eliminate free pixels.
                    #    A 'free' pixel is a polygon with area less than min_poly_area, and
                    #    more than max_poly_dist away from its closest neighbor.

                    min_poly_area = 5.5 # Hardcoded magic number, will change with resolution/upsampling
                    max_poly_dist = max(p_s_x, p_s_y)

                    # We find all polys smaller than our cutoff, and measure the minimum distance between them and any
                    # other poly in this ROI (that is not them)

                    small_polys = [(poly, min([poly.distance(cand_poly) for cand_poly in contour_list_roi])) for poly in contour_list_roi if poly.area < min_poly_area]
                    big_polys = [poly for poly in contour_list_roi if poly.area >= min_poly_area]

                    for poly, min_dist in small_polys:
                        if min_dist > 0.0 and min_dist < max_poly_dist:
                            big_polys.append(poly) # A small poly that is closer than max_poly_dist isn't really small.

                    postprocessed_roi_list = big_polys

                    if not postprocessed_roi_list:
                        continue # Don't add anything if there aren't contours left

                    # Step 2: Close gaps. We dilate and erode polygons using buffers.

                    buffer_radius = 5*max_poly_dist
                    current_z = postprocessed_roi_list[0].exterior.coords[0][2]
                    buffer_poly = MultiPolygon(postprocessed_roi_list).buffer(buffer_radius).buffer(-buffer_radius)

                    if isinstance(buffer_poly, MultiPolygon):
                        postprocessed_roi_list = [Polygon([(p[0], p[1], current_z) for p in poly.exterior.coords[:]]) for poly in buffer_poly]
                    elif isinstance(buffer_poly, Polygon):
                        postprocessed_roi_list = [Polygon([(p[0], p[1], current_z) for p in buffer_poly.exterior.coords[:]])]
                    if not postprocessed_roi_list:
                        continue # Don't add anything if there aren't contours left

                    # Step 3: ROI-specific contour number restriction
                    if roi_name == 'RtLung' or roi_name == 'LtLung':
                        # Up to 3 lung contours allowed - take biggest by area
                        if len(postprocessed_roi_list) > 3:
                            postprocessed_roi_list = sorted(postprocessed_roi_list, key=lambda x: x.area, reverse=True)[0:3]
                    else:
                        # Up to 1 contour allowed for most structures - take biggest by area
                        if len(postprocessed_roi_list) > 1:
                            postprocessed_roi_list = [sorted(postprocessed_roi_list, key=lambda x: x.area, reverse=True)[0]]

                # Set list of geometries for this ROI
                contour_list_pred[roi_name] = postprocessed_roi_list

        #Additional postprocessing based on ROI-on-ROI interactions goes here
        output_dict = {}
        for k,v in contour_list_pred.iteritems():
            output_dict[k] = [poly.exterior.coords[:] for poly in v]

        return output_dict

    except:
        print('Error in threaded op')

# Can likely remove most, but will check first
def __predict_slices(images, model, mean_pixel, class_list):
    """
    Core prediction function. Uses the Keras model in GPU mode to generate probability maps for each slice, then
    uses Shapely geometry processing with multiprocessing to convert those raster maps to vector data.
    :param images: A list of pydicom Datasets representing the CT slices to predict contours for
    :param model: A pre-trained and initialized Beaunet Keras model
    :param class_list: A list of contour names in order describing the output vector of the Keras model
    :return: A dictionary mapping slice UIDs to ROI names to a list of contours for that ROI.
    """

    # Process images in sequence
    slices_to_contours = {}

    executor = concurrent.futures.ProcessPoolExecutor(24)
    futures = [executor.submit(__threaded_geom_op, item) for item in zip(images, y_data, repeat(class_list, len(images)))]
    concurrent.futures.wait(futures)
    results = [output.result() for output in futures]

    for image_ds, predicted_contours in zip(images, results):
        slices_to_contours[image_ds.SOPInstanceUID] = {}
        for roi_name, contour_list in predicted_contours.iteritems():
            slices_to_contours[image_ds.SOPInstanceUID][roi_name] = contour_list

    # Return mapping of slices to contours and ROI names to slices/contours
    return slices_to_contours



def main():
    # Check that input directory exists and is a directory
    if not os.path.isdir(INPUT_PATH):
        print('Input path \'{0:s}\' does not exist or is not a directory'.format(INPUT_PATH))
        exit(-1)
    
    # Check that stats directory exists
    if not os.path.isdir(STATS_PATH):
        print('Stats path \'{0:s}\' does not exist or is not a directory'.format(STATS_PATH))
        exit(-1)

    # Check that pixel stats file exists
    if not os.path.exists(os.path.join(STATS_PATH, 'pixel_stats.txt')):
        print('Pixel stats file does not exist as \'{0:s}\''.format(os.path.join(STATS_PATH, 'pixel_stats.txt')))
        exit(-1)

    mean_pixel = 0.0

    with open(os.path.join(STATS_PATH, 'pixel_stats.txt'), 'r') as pixel_stats_file:
        for line in pixel_stats_file:
            key, values = line.split('\t')

            if key.strip() == 'mean_pixel':
                mean_pixel = float(values.strip())

    # Check that color dictionary exists
    if not os.path.exists(os.path.join(STATS_PATH, 'color_dictionary.txt')):
        print('Color dictionary file does not exist as \'{0:s}\''.format(os.path.join(STATS_PATH, 'color_dictionary.txt')))
        exit(-1)

    color_dict = {}
    class_list = []

    with open(os.path.join(STATS_PATH, 'color_dictionary.txt')) as color_dict_file:
        color_dict['background'] = (0, 0, 0, 0)
        class_list.append('background')
        for line in color_dict_file:
            roi, r, g, b = line.split('\t')
            color_dict[roi] = (len(color_dict), int(r), int(g), int(b))
            class_list.append(roi)
    
    SERIES = {}

    if HIERARCHICAL_MODE:
        # Scan through directory tree for studies and create listing
        for dirpath, _, filenames in os.walk(INPUT_PATH):
            filtered_filenames = [f for f in filenames if os.path.splitext(f)[1].lower() == '.dcm']
            for filename in filtered_filenames:
                ds = read_file(os.path.join(dirpath, filename), defer_size="4KB", stop_before_pixels=True)
                if ds.SOPClassUID == 'CT Image Storage':
                    if ds.SeriesInstanceUID not in SERIES:
                        SERIES[ds.SeriesInstanceUID] = {'path':dirpath, 'slices':[]}
                    SERIES[ds.SeriesInstanceUID]['slices'].append(filename)

    else:
        filenames = [f for f in os.listdir(INPUT_PATH) if os.path.isfile(os.path.join(INPUT_PATH,f)) and
                     os.path.splitext(f)[1].lower() == '.dcm']
        for filename in filenames:
            ds = read_file(os.path.join(INPUT_PATH, filename), defer_size="4KB", stop_before_pixels=True)
            if ds.SOPClassUID == 'CT Image Storage':
                if ds.SeriesInstanceUID not in SERIES:
                    SERIES[ds.SeriesInstanceUID] = {'path': INPUT_PATH, 'slices': []}
                SERIES[ds.SeriesInstanceUID]['slices'].append(filename)

    # Process series
    for image_series in SERIES.itervalues():
        dirpath = image_series['path']
        filenames = image_series['slices']
        if HIERARCHICAL_MODE:
            print('  Processing \'{0:s}\''.format(dirpath))
            print('  Output to \'{0:s}\''.format(OUTPUT_FILENAME))

        start_time = time.time()
        # Start crafting the RTSTRUCT
        rtstruct_ds = dataset.Dataset()
        current_time = time.localtime()
        rtstruct_ds.SpecificCharacterSet = 'ISO_IR 100'
        rtstruct_ds.InstanceCreationDate = time.strftime('%Y%m%d', current_time) # DICOM date format
        rtstruct_ds.InstanceCreationTime = time.strftime('%H%M%S.%f', current_time)[:-3] # DICOM time format
        rtstruct_ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # RT Structure Set Storage Class
        rtstruct_ds.SOPInstanceUID = uid.generate_uid()
        rtstruct_ds.Modality = 'RTSTRUCT'
        rtstruct_ds.Manufacturer = 'Beaumont Health'
        rtstruct_ds.ManufacturersModelName = 'Beaunet Automatic Segmentation Engine'
        rtstruct_ds.SoftwareVersions = ['0.8.3']
        rtstruct_ds.StructureSetDate = time.strftime('%Y%m%d', current_time)
        rtstruct_ds.StructureSetTime = time.strftime('%H%M%S.%f', current_time)[:-3]
        rtstruct_ds.StructureSetLabel = 'Beaunet Auto-Segmented Contours'
        rtstruct_ds.StructureSetName = 'Beaunet Auto-Segmented Contours'

        # Scan through input directory and grab common data; first file will be used to initialize demographics
        # and study-level variables
        rtstruct_init = False
        contour_image_uids = {}
        roi_images = {}

        #for image_file_name in next(os.walk(INPUT_PATH))[2]:
        for image_file_name in filenames:
            # Don't try to force a read; instead, just attempt, and if it doesn't work,
            # or if this is not a CT slice, go to the next image.
            image_ds = read_file(os.path.join(dirpath, image_file_name), force=False)
            if not image_ds.SOPClassUID == uid.UID('1.2.840.10008.5.1.4.1.1.2'):
                continue
            if not rtstruct_init:
                # Referenced study
                referenced_study_ds = dataset.Dataset()
                referenced_study_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.3.1.2.3.2') # Study Component Management SOP Class
                referenced_study_ds.ReferencedSOPInstanceUID = image_ds.StudyInstanceUID
                rtstruct_ds.ReferencedStudySequence = sequence.Sequence([referenced_study_ds])
                # Demographics
                if 'PatientsName' in image_ds:
                    rtstruct_ds.PatientsName = image_ds.PatientsName
                else:
                    rtstruct_ds.PatientsName = 'UNKNOWN^UNKNOWN^^'
                if 'PatientID' in image_ds:
                    rtstruct_ds.PatientID = image_ds.PatientID
                else:
                    rtstruct_ds.PatientID = '0000000'

                if 'PatientsBirthDate' in image_ds:
                    rtstruct_ds.PatientsBirthDate = image_ds.PatientsBirthDate
                else:
                    rtstruct_ds.PatientsBirthDate = ''

                if 'PatientsSex' in image_ds:
                    rtstruct_ds.PatientsSex = image_ds.PatientsSex
                else:
                    rtstruct_ds.PatientsSex = ''

                # This study
                rtstruct_ds.StudyInstanceUID = image_ds.StudyInstanceUID
                rtstruct_ds.SeriesInstanceUID = uid.generate_uid()

                if 'StudyID' in image_ds:
                    rtstruct_ds.StudyID = image_ds.StudyID
                if 'SeriesNumber' in image_ds:
                    rtstruct_ds.SeriesNumber = image_ds.SeriesNumber
                if 'StudyDate' in image_ds:
                    rtstruct_ds.StudyDate = image_ds.StudyDate
                if 'StudyTime' in image_ds:
                    rtstruct_ds.StudyTime = image_ds.StudyTime

                # Referenced frame of reference
                referenced_frame_of_reference_ds = dataset.Dataset()
                rtstruct_ds.ReferencedFrameOfReferenceSequence = sequence.Sequence([referenced_frame_of_reference_ds])
                referenced_frame_of_reference_ds.FrameOfReferenceUID = image_ds.FrameOfReferenceUID
                rt_referenced_study_ds = dataset.Dataset()
                referenced_frame_of_reference_ds.RTReferencedStudySequence = sequence.Sequence([rt_referenced_study_ds])
                rt_referenced_study_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.3.1.2.3.2') # Study Component Management SOP Class
                rt_referenced_study_ds.ReferencedSOPInstanceUID = image_ds.StudyInstanceUID
                rt_referenced_series_ds = dataset.Dataset()
                rt_referenced_study_ds.RTReferencedSeriesSequence = sequence.Sequence([rt_referenced_series_ds])
                rt_referenced_series_ds.SeriesInstanceUID = image_ds.SeriesInstanceUID
                rt_referenced_series_ds.ContourImageSequence = sequence.Sequence()

                # Once all is done, mark initialized
                rtstruct_init = True

            # We only store the final contourimagesequence after all is done to maintain ordering
            contour_image_ds = dataset.Dataset()
            contour_image_ds.ReferencedSOPClassUID = uid.UID('1.2.840.10008.5.1.4.1.1.2') # CT Image Storage
            contour_image_ds.ReferencedSOPInstanceUID = image_ds.SOPInstanceUID
            contour_image_uids[image_ds.SOPInstanceUID] = {'ds':contour_image_ds, 'filename':image_file_name, 'key': image_ds.SOPInstanceUID}

        # All images scanned (not processed!), build more sequences
        # This is all the images in the referenced series, regardless of whether they have contours or not
        rt_referenced_series_ds.ContourImageSequence.extend([contour_image_uids[key]['ds'] for key in sorted(contour_image_uids.keys())])

        # Create sequence objects for contours - these may be empty if no contours are created
        rtstruct_ds.StructureSetROISequence = sequence.Sequence()
        rtstruct_ds.ROIContourSequence = sequence.Sequence()
        rtstruct_ds.RTROIObservationsSequence = sequence.Sequence()

        # Process the images one by one in order of contour sequence.
        image_num = 0
        image_batch = []
        image_processing_order = sorted(contour_image_uids.keys())

        for image_file_name, _ in [(contour_image_uids[key]['filename'], contour_image_uids[key]['key']) for key in image_processing_order]:
            try:
                image_ds = read_file(os.path.join(dirpath, image_file_name))
                # Do work if this DICOM file is a CT slice
                if image_ds.SOPClassUID == uid.UID('1.2.840.10008.5.1.4.1.1.2'):
                    if len(image_batch) >= 24:
                        predict_slices_contours = __predict_slices(image_batch, model, mean_pixel, class_list)
                        # Add predictions to general list
                        for slice_id, contour_dict in predict_slices_contours.iteritems():
                            contour_image_uids[slice_id]['contours'] = {}
                            for roi_name, contour_list in contour_dict.iteritems():
                                contour_image_uids[slice_id]['contours'][roi_name] = contour_list
                                if roi_name not in roi_images:
                                    roi_images[roi_name] = []
                                roi_images[roi_name].append([slice_id, contour_list])
                        image_num += len(image_batch)
                        print('    Processed slices {0:d} of {1:d}'.format(image_num, len(image_processing_order)))
                        image_batch = []
                    image_batch.append(image_ds)
            except Exception as e:
                print('Exception: {0:s}'.format(e.message))

        predict_slices_contours = __predict_slices(image_batch, model, mean_pixel, class_list)
        # Add predictions to general list

        for slice_id, contour_dict in predict_slices_contours.iteritems():
            contour_image_uids[slice_id]['contours'] = {}
            for roi_name, contour_list in contour_dict.iteritems():
                contour_image_uids[slice_id]['contours'][roi_name] = contour_list
                if roi_name not in roi_images:
                    roi_images[roi_name] = []
                    roi_images[roi_name].append([slice_id, contour_list])
            image_num += len(image_batch)
            print('    Processed slices {0:d} of {1:d}'.format(image_num, len(image_processing_order)))

            # Now we have contours for each CT slice. We have a mapping of CT slice ID to data:
            #   contour_image_uids[ID] = {ds: image_ds, filename: str, key: ID, contours: {roi_name: list}}
            # And a mapping of ROI name to slices:
            #   roi_images[roi_name] = [[ID, contourlist], ...]
            # Let's construct the tags.

            rtstruct_ds.StructureSetROISequence = sequence.Sequence([])
            rtstruct_ds.ROIContourSequence = sequence.Sequence([])
            rtstruct_ds.RTROIObservationsSequence = sequence.Sequence([])
            roi_number = 1

            for roi_name in class_list:
                # If the ROI exists in our image set...
                if roi_name in roi_images:
                    # Add ROI to Structure Set Sequence
                    structure_set_roi_ds = dataset.Dataset()
                    rtstruct_ds.StructureSetROISequence.append(structure_set_roi_ds)
                    structure_set_roi_ds.ROINumber = roi_number
                    structure_set_roi_ds.ReferencedFrameOfReferenceUID = rtstruct_ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                    structure_set_roi_ds.ROIName = roi_name
                    structure_set_roi_ds.ROIGenerationAlgorithm = 'AUTOMATIC'

                    # Add ROI to ROI Contour Sequence
                    roi_contour_ds = dataset.Dataset()
                    rtstruct_ds.ROIContourSequence.append(roi_contour_ds)
                    roi_contour_ds.ROIDisplayColor = list(color_dict[roi_name][1:])
                    roi_contour_ds.ReferencedROINumber = roi_number
                    roi_contour_ds.ContourSequence = sequence.Sequence([])

                    for slice_id, contour_list in roi_images[roi_name]:
                        for contour in contour_list:
                            contour_ds = dataset.Dataset()
                            contour_ds.ContourImageSequence = sequence.Sequence([contour_image_uids[slice_id]['ds']])
                            contour_ds.ContourGeometricType = 'CLOSED_PLANAR'
                            contour_ds.NumberOfContourPoints = len(contour)
                            contour_ds.ContourData = ['{0:0.2f}'.format(val) for p in contour for val in p]
                            roi_contour_ds.ContourSequence.append(contour_ds)

                    # Add ROI to RT ROI Observations Sequence
                    rt_roi_obs = dataset.Dataset()
                    rtstruct_ds.RTROIObservationsSequence.append(rt_roi_obs)
                    rt_roi_obs.ObservationNumber = roi_number
                    rt_roi_obs.ReferencedROINumber = roi_number
                    rt_roi_obs.ROIObservationLabel = roi_name
                    rt_roi_obs.RTROIInterpretedType = 'ORGAN'

                    # Update ROI number
                    roi_number += 1

            # RTSTRUCT creation complete. Write to output.
            if HIERARCHICAL_MODE:
                output_path = os.path.join(dirpath, OUTPUT_FILENAME)
            else:
                output_path = OUTPUT_FILENAME

            file_meta = dataset.Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            file_meta.MediaStorageSOPInstanceUID = uid.generate_uid()
            file_meta.ImplementationClassUID = uid.generate_uid()

            output_ds = dataset.FileDataset(output_path, {}, file_meta=file_meta, preamble="\0" * 128)
            output_ds.update(rtstruct_ds)
            output_ds.save_as(output_path)
            end_time = time.time()

        # Write profiling to console
        seconds = int(end_time - start_time)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        if days > 0:
            time_elapsed = "{0:d}d, {1:d}h, {2:d}m, {3:d}s".format(days, hours, minutes, seconds)
        elif hours > 0:
            time_elapsed = "{0:d}h, {1:d}m, {2:d}s".format(hours, minutes, seconds)
        elif minutes > 0:
            time_elapsed = "{0:d}m, {1:d}s".format(minutes, seconds)
        else:
            time_elapsed = "{0:d}s".format(seconds)

        print('    Completed run in {0:s}'.format(time_elapsed))

if __name__ == '__main__':
    main()