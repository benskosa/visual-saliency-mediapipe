"""
Serializer for the recognition result using protobuf
"""

from typing import List, Tuple, Dict, Any, Union
from .recognition_data_pb2 import RecognitionData, MaskContour, ContourPoint, Box, GeometryCenter, Mask, InnerMask, MaskColor, Vec3, Quat, LabelProperty

def serialize(
        result: Dict[str, Any],
        colors: Union[Tuple, Dict[str, Tuple], None] = None,
        augmentations: Union[Dict[str, str], str, None] = None,
        thicknesses: Union[Dict[str, int], float, None] = None,
        label_colors: Union[Dict[str, Tuple], Tuple, None] = None,
        timestamp: int = 0, # timestamp is uint64, use 0 for invalid,
        name_mapping: Dict[str, str] = None,
        ) -> str:
    """
    Serialize the recognition result to a protobuf message
    Args:
        result: The recognition result to be serialized
    Returns:
        The serialized recognition result
    """
    rdata = RecognitionData()

    # the fields are all optional, so we need to check if the field exists before serializing it
    _serilizer = {
        'masks': _serialize_mask,
        'mask_contours': _serialize_mask_contour,
        'boxes': _serialize_box,
        'geometry_center': _serialize_geometry_center,
        'scores': _serialize_scores,
        'class_names': _serialize_class_names,
        # 'position': _serialize_position,
        # 'rotation': _serialize_rotation,
        # colors and augmentations are special cases
    }

    for key, value in result.items():
        if key in _serilizer:
            _serilizer[key](value, rdata, name_mapping=name_mapping)

    # if both position and rotation are provided, set the pose_valid flag to True
    # if 'position' in result and 'rotation' in result:
    #     rdata.pose_valid = True
    # else:
    #     rdata.pose_valid = False

    # set the timestamp
    rdata.timestamp = timestamp

    # if colors is provided, serialize it
    # if it's a tuple, then it's a single color for all masks
    # if it's a dict, then it's a color for each class
    if colors is not None:
        if isinstance(colors, tuple):
            # pad it to a list
            colors = [colors for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the color
            class_names = result['class_names']
            colors = [colors[class_name] for class_name in class_names]
        _serialize_colors(colors, rdata, field='color')

    # if label colors is provided, serialize it
    # if it's a tuple, then it's a single color for all labels
    # if it's a dict, then it's a color for each class
    if label_colors is not None:
        if isinstance(label_colors, tuple):
            # pad it to a list
            label_colors = [label_colors for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the color
            class_names = result['class_names']
            label_colors = [label_colors[class_name] for class_name in class_names]
        _serialize_colors(label_colors, rdata, field='label_color')

    # if augmentations is provided, serialize it
    # if it's a string, then it's a single augmentation for all objects
    # if it's a dict, then it's an augmentation for each class
    if augmentations is not None:
        if isinstance(augmentations, str):
            # pad it to a list
            augmentations = [augmentations for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the augmentation
            class_names = result['class_names']
            augmentations = [augmentations[class_name] for class_name in class_names]
        _serialize_augmentations(augmentations, rdata)

    # if thicknesses is provided, serialize it
    # if it's a float, then it's a single thickness for all objects
    # if it's a dict, then it's a thickness for each class
    if thicknesses is not None:
        if not isinstance(thicknesses, dict):
            # pad it to a list
            thicknesses = [thicknesses for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the thickness
            class_names = result['class_names']
            thicknesses = [thicknesses[class_name] for class_name in class_names]
        _serialize_thicknesses(thicknesses, rdata)

    # if text color, label size, or text size are provided
    if 'text_color' in result or 'label_size' in result or 'text_size' in result:
        label_property = LabelProperty()

        if 'text_color' in result:
            label_property.text_color.r = result['text_color'][0]
            label_property.text_color.g = result['text_color'][1]
            label_property.text_color.b = result['text_color'][2]
            # if alpha is provided, set it
            if len(result['text_color']) == 4:
                label_property.text_color.a = result['text_color'][3]

        if 'label_size' in result:
            label_property.label_size = result['label_size']

        if 'text_size' in result:
            label_property.text_size = result['text_size']

        rdata.label_property.CopyFrom(label_property)

    return rdata.SerializeToString()


def serialize_face(
        result: Dict[str, Any],
        colors: Union[Tuple, Dict[str, Tuple], None] = None,
        augmentations: Union[Dict[str, str], str, None] = None,
        thicknesses: Union[Dict[str, int], float, None] = None,
        timestamp: int = 0, # timestamp is uint64, use 0 for invalid,
        name_mapping: Dict[str, str] = None,
        ) -> str:
    """
    Serialize the recognition result FOR face_landmarks to a protobuf message
    Args:
        result: The recognition result to be serialized
    Returns:
        The serialized recognition result
    """
    rdata = RecognitionData()

    # the fields are all optional, so we need to check if the field exists before serializing it
    _serilizer = {
        'masks': _serialize_mask,
        'mask_contours': _serialize_mask_contour,
        'boxes': _serialize_box,
        'geometry_center': _serialize_geometry_center,
        'scores': _serialize_scores,
        'class_names': _serialize_class_names,
        # 'position': _serialize_position,
        # 'rotation': _serialize_rotation,
        # colors and augmentations are special cases
    }

    for key, value in result.items():
        if key in _serilizer:
            _serilizer[key](value, rdata, name_mapping=name_mapping)

    # if both position and rotation are provided, set the pose_valid flag to True
    # if 'position' in result and 'rotation' in result:
    #     rdata.pose_valid = True
    # else:
    #     rdata.pose_valid = False

    # set the timestamp
    rdata.timestamp = timestamp

    # if colors is provided, serialize it
    # if it's a tuple, then it's a single color for all masks
    # if it's a dict, then it's a color for each class
    if colors is not None:
        if isinstance(colors, tuple):
            # pad it to a list
            colors = [colors for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the color
            class_names = result['class_names']
            colors = [colors[class_name] for class_name in class_names]
        _serialize_colors(colors, rdata, field='color')

    # if label colors is provided, serialize it
    # if it's a tuple, then it's a single color for all labels
    # if it's a dict, then it's a color for each class
    if label_colors is not None:
        if isinstance(label_colors, tuple):
            # pad it to a list
            label_colors = [label_colors for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the color
            class_names = result['class_names']
            label_colors = [label_colors[class_name] for class_name in class_names]
        _serialize_colors(label_colors, rdata, field='label_color')

    # if augmentations is provided, serialize it
    # if it's a string, then it's a single augmentation for all objects
    # if it's a dict, then it's an augmentation for each class
    if augmentations is not None:
        if isinstance(augmentations, str):
            # pad it to a list
            augmentations = [augmentations for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the augmentation
            class_names = result['class_names']
            augmentations = [augmentations[class_name] for class_name in class_names]
        _serialize_augmentations(augmentations, rdata)

    # if thicknesses is provided, serialize it
    # if it's a float, then it's a single thickness for all objects
    # if it's a dict, then it's a thickness for each class
    if thicknesses is not None:
        if not isinstance(thicknesses, dict):
            # pad it to a list
            thicknesses = [thicknesses for _ in range(len(result['class_names']))]
        else:
            # based on the class name, get the thickness
            class_names = result['class_names']
            thicknesses = [thicknesses[class_name] for class_name in class_names]
        _serialize_thicknesses(thicknesses, rdata)

    # if text color, label size, or text size are provided
    if 'text_color' in result or 'label_size' in result or 'text_size' in result:
        label_property = LabelProperty()

        if 'text_color' in result:
            label_property.text_color.r = result['text_color'][0]
            label_property.text_color.g = result['text_color'][1]
            label_property.text_color.b = result['text_color'][2]
            # if alpha is provided, set it
            if len(result['text_color']) == 4:
                label_property.text_color.a = result['text_color'][3]

        if 'label_size' in result:
            label_property.label_size = result['label_size']

        if 'text_size' in result:
            label_property.text_size = result['text_size']

        rdata.label_property.CopyFrom(label_property)

    return rdata.SerializeToString()


def _serialize_mask(masks: List[List[List[int]]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize masks to protobuf message, and add it to the recognition data
    Args:
        mask: The mask to be serialized, in the format of [[[x11, x12, x13], ...], ...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    Note: the desired mask is a bool mask, but the parameter is a int mask,
        so we need to convert it to a bool mask
    """
    for mask in masks:
        pmask = Mask()
        for inner_mask in mask:
            pinner_mask = InnerMask()
            pinner_mask.mask.extend([bool(pixel) for pixel in inner_mask])
            pmask.inner_masks.append(pinner_mask)
        rdata.masks.append(pmask)

def _serialize_mask_contour(mask_contour: List[List[Tuple[int, int]]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize contours to protobuf message, and add it to the recognition data
    Args:
        mask_contour: The contour to be serialized, in the format of [(x11, y11), ...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for contour in mask_contour:
        pmask_contour = MaskContour()
        for point in contour:
            ppoint = ContourPoint()
            ppoint.x = point[0]
            ppoint.y = point[1]
            pmask_contour.inner_contours.append(ppoint)
        rdata.mask_contours.append(pmask_contour)

def _serialize_box(boxes: List[List[int]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize boxes to protobuf message, and add it to the recognition data
    Args:
        box: The box to be serialized, in the format of [x1, y1, x2, y2]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for box in boxes:
        pbox = Box()
        pbox.x1 = box[0]
        pbox.y1 = box[1]
        pbox.x2 = box[2]
        pbox.y2 = box[3]
        rdata.boxes.append(pbox)

def _serialize_geometry_center(geometry_center: List[Tuple[int, int]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize geometry centers to protobuf message, and add it to the recognition data
    Args:
        geometry_center: The geometry centers to be serialized, in the format of [(x1, y1), ...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for center in geometry_center:
        pgeometry_center = GeometryCenter()
        pgeometry_center.x = center[0]
        pgeometry_center.y = center[1]
        rdata.geometry_centers.append(pgeometry_center)

def _serialize_scores(scores: List[float], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize scores to protobuf message, and add it to the recognition data
    Args:
        scores: The scores to be serialized
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    rdata.scores.extend(scores)

def _serialize_class_names(class_names: List[str], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize class names to protobuf message, and add it to the recognition data
    Args:
        class_names: The class names to be serialized
        rdata: The protobuf message to be serialized to
        kwargs: name_mapping: The mapping from class names to display names
    Returns:
        None
    """
    name_mapping = kwargs.get('name_mapping', None)
    if name_mapping is not None:
        class_names = [name_mapping.get(name, name) for name in class_names]
    rdata.class_names.extend(class_names)

def _serialize_augmentations(augmentations: List[str], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize augmentations to protobuf message, and add it to the recognition data
    Args:
        augmentations: The augmentations to be serialized
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    rdata.augmentations.extend(augmentations)

def _serialize_thicknesses(thicknesses: List[float], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize thicknesses to protobuf message, and add it to the recognition data
    Args:
        thicknesses: The thicknesses to be serialized
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    rdata.contour_thicknesses.extend(thicknesses)

def _serialize_colors(colors: List[List[int]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize colors to protobuf message, and add it to the recognition data
    Args:
        colors: The colors to be serialized, in the format of [[r1, g1, b1], ...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    field = kwargs.get('field', 'color')
    if field == 'color':
        to_append = rdata.colors
    elif field == 'label_color':
        to_append = rdata.label_colors
    else:
        raise ValueError(f"Unknown field: {field}")
    
    for color in colors:
        pcolor = MaskColor()
        pcolor.r = color[0]
        pcolor.g = color[1]
        pcolor.b = color[2]
        # if alpha is provided, set it
        if len(color) == 4:
            pcolor.a = color[3]
        to_append.append(pcolor)

def _serialize_position(position: List[float], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize position to protobuf message, and add it to the recognition data
    Args:
        position: The position to be serialized, in the format of [x, y, z]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    pposition = Vec3()
    pposition.x = position[0]
    pposition.y = position[1]
    pposition.z = position[2]
    rdata.camera_position.CopyFrom(pposition)

def _serialize_rotation(rotation: List[float], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize rotation to protobuf message, and add it to the recognition data
    Args:
        rotation: The rotation to be serialized, in the format of [x, y, z, w]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    # protation = Quat()
    # protation.x = rotation[0]
    # protation.y = rotation[1]
    # protation.z = rotation[2]
    # protation.w = rotation[3]
    
    protation = Vec3()
    protation.x = rotation[0]
    protation.y = rotation[1]
    protation.z = rotation[2]
    rdata.camera_rotation.CopyFrom(protation)