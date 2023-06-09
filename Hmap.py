from random import randint
from socket import gethostname
from typing import Tuple
import cv2
import highdicom as hd
import numpy as np
import pydicom
from pydicom.sr.codedict import codes
from pydicom import Dataset


def parametric_map(
    source_image: pydicom.Dataset,
    input_map: np.ndarray,
    class_label: str,
    manufacturer: str,
    manufacturer_model_name: str,
):
    # Encode pixel data as unsigned integers for improved interoperability
    pixel_data_bits = 8
    dtype = np.dtype(f"uint{pixel_data_bits}")
    slope = 2.0 / (2**pixel_data_bits - 1)
    intercept = -1.0
    rescaled_map = ((input_map - intercept) / slope).astype(dtype)

    rows = source_image.TotalPixelMatrixRows
    cols = source_image.TotalPixelMatrixColumns

    pixel_array = np.zeros((1, rows, cols, 1), dtype=dtype)
    print(pixel_array.shape)
    resized_plane = cv2.resize(rescaled_map,dsize=(cols, rows),interpolation=cv2.INTER_CUBIC,)
    pixel_array = resized_plane
    print(pixel_array.shape)

    value_range = [pixel_array.min(), pixel_array.max()]
    window_width = value_range[1] - value_range[0]
    # Make sure this is an integer
    window_center = int(int(value_range[1]) + int(value_range[0]) // 2)

    real_world_value_mappings = [
        hd.pm.RealWorldValueMapping(
            lut_label="0",
            lut_explanation=class_label,
            unit=codes.UCUM.NoUnits,
            value_range=value_range,
            intercept=intercept,
            slope=slope,
        )
    ]
    plane_position, pixel_measures = _compute_derived_image_attributes(source_image, pixel_array)

    parametric_map = hd.pm.ParametricMap(
        [source_image],
        pixel_array=pixel_array,
        series_instance_uid=hd.UID(),
        series_number=randint(1, 100),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        software_versions="0.0.1",
        device_serial_number=f"{gethostname()}",
        contains_recognizable_visual_features=False,
        real_world_value_mappings=real_world_value_mappings,
        window_center=window_center,
        window_width=window_width,
        plane_positions=[plane_position],
        pixel_measures=pixel_measures,
        content_label="HEATMAP",
    )

    return parametric_map




def _compute_derived_image_attributes(source_image: Dataset, total_pixel_matrix: np.ndarray) -> Tuple[hd.PlanePositionSequence, hd.PixelMeasuresSequence]:
    """Compute attributes of a derived single-frame image.
    Parameters
    ----------
    source_image: pydicom.Dataset
        Source image from which single-frame image was derived
    total_pixel_matrix: numpy.ndarray
        Total Pixel matrix of derived single-frame image for which attribute
        values should be computed
    Returns
    -------
    plane_positions: highdicom.PlanePositionSequence
        Plane position of the single-frame image
    pixel_measures: highdicom.PixelMeasuresSequence
        Pixel measures of the single-frame image
    """
    sm_total_rows = int(
        np.ceil(source_image.TotalPixelMatrixRows / source_image.Rows)
        * source_image.Rows
    )
    sm_total_cols = int(
        np.ceil(source_image.TotalPixelMatrixColumns / source_image.Columns)
        * source_image.Columns
    )
    origin = source_image.TotalPixelMatrixOriginSequence[0]
    x_offset = origin.XOffsetInSlideCoordinateSystem
    y_offset = origin.YOffsetInSlideCoordinateSystem
    sm_shared_func_groups = source_image.SharedFunctionalGroupsSequence[0]
    sm_pixel_measures = sm_shared_func_groups.PixelMeasuresSequence[0]
    sm_pixel_spacing = sm_pixel_measures.PixelSpacing
    sm_slice_thickness = sm_pixel_measures.SliceThickness
    derived_pixel_spacing = (
        (sm_total_rows * sm_pixel_spacing[0]) / total_pixel_matrix.shape[0],
        (sm_total_cols * sm_pixel_spacing[1]) / total_pixel_matrix.shape[1],
    )
    derived_plane_position = hd.PlanePositionSequence(
        coordinate_system=hd.CoordinateSystemNames.SLIDE,
        image_position=(x_offset, y_offset, 0.0),
        pixel_matrix_position=(1, 1),  # there is only one frame
    )
    derived_pixel_measures = hd.PixelMeasuresSequence(
        pixel_spacing=derived_pixel_spacing, slice_thickness=sm_slice_thickness
    )
    return (derived_plane_position, derived_pixel_measures)





dicomFile_Kidney=pydicom.dcmread('./Data_cohorts/IDC-CPTAC-LUAD/5b5b7f0e-5c8f-4f15-ac9e-58cb649c6ad8.dcm')
Kidneymap=np.zeros((1000,1000),dtype=np.uint8)
Kidneymap[10:50,50:160]=1
PMap=parametric_map(dicomFile_Kidney,Kidneymap,class_label='1',manufacturer='me',manufacturer_model_name='ubuntu')
PMap.save_as('/home/m813r/Documents/Data_cohorts/Test_Data_Patho_Dicom/Kidney/map.dcm',write_like_original=False)
plane_position, pixel_measures = _compute_derived_image_attributes(dicomFile_Kidney, Kidneymap)

