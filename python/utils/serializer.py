"""
Serializer for the recognition result using protobuf
"""

from typing import List, Tuple, Dict, Any, Union
from .recognition_data_face_pb2 import RecognitionData, NormalizedLandmarkList, NormalizedLandmark, FaceMeshColors, EdgeColor, FaceAugmentations, FaceMeshThicknesses, Box

# def serialize(
#         result: Dict[str, Any],
#         colors: Union[Tuple, Dict[str, Tuple], None] = None,
#         augmentations: Union[Dict[str, str], str, None] = None,
#         thicknesses: Union[Dict[str, int], float, None] = None,
#         label_colors: Union[Dict[str, Tuple], Tuple, None] = None,
#         timestamp: int = 0, # timestamp is uint64, use 0 for invalid,
#         name_mapping: Dict[str, str] = None,
#         ) -> str:
#     """
#     Serialize the recognition result to a protobuf message
#     Args:
#         result: The recognition result to be serialized
#     Returns:
#         The serialized recognition result
#     """
#     rdata = RecognitionData()

#     # the fields are all optional, so we need to check if the field exists before serializing it
#     _serilizer = {
#         'masks': _serialize_mask,
#         'mask_contours': _serialize_mask_contour,
#         'boxes': _serialize_box,
#         'geometry_center': _serialize_geometry_center,
#         'scores': _serialize_scores,
#         'class_names': _serialize_class_names,
#         # 'position': _serialize_position,
#         # 'rotation': _serialize_rotation,
#         # colors and augmentations are special cases
#     }

#     for key, value in result.items():
#         if key in _serilizer:
#             _serilizer[key](value, rdata, name_mapping=name_mapping)

#     # if both position and rotation are provided, set the pose_valid flag to True
#     # if 'position' in result and 'rotation' in result:
#     #     rdata.pose_valid = True
#     # else:
#     #     rdata.pose_valid = False

#     # set the timestamp
#     rdata.timestamp = timestamp

#     # if colors is provided, serialize it
#     # if it's a tuple, then it's a single color for all masks
#     # if it's a dict, then it's a color for each class
#     if colors is not None:
#         if isinstance(colors, tuple):
#             # pad it to a list
#             colors = [colors for _ in range(len(result['class_names']))]
#         else:
#             # based on the class name, get the color
#             class_names = result['class_names']
#             colors = [colors[class_name] for class_name in class_names]
#         _serialize_colors(colors, rdata, field='color')

#     # if label colors is provided, serialize it
#     # if it's a tuple, then it's a single color for all labels
#     # if it's a dict, then it's a color for each class
#     if label_colors is not None:
#         if isinstance(label_colors, tuple):
#             # pad it to a list
#             label_colors = [label_colors for _ in range(len(result['class_names']))]
#         else:
#             # based on the class name, get the color
#             class_names = result['class_names']
#             label_colors = [label_colors[class_name] for class_name in class_names]
#         _serialize_colors(label_colors, rdata, field='label_color')

#     # if augmentations is provided, serialize it
#     # if it's a string, then it's a single augmentation for all objects
#     # if it's a dict, then it's an augmentation for each class
#     if augmentations is not None:
#         if isinstance(augmentations, str):
#             # pad it to a list
#             augmentations = [augmentations for _ in range(len(result['class_names']))]
#         else:
#             # based on the class name, get the augmentation
#             class_names = result['class_names']
#             augmentations = [augmentations[class_name] for class_name in class_names]
#         _serialize_augmentations(augmentations, rdata)

#     # if thicknesses is provided, serialize it
#     # if it's a float, then it's a single thickness for all objects
#     # if it's a dict, then it's a thickness for each class
#     if thicknesses is not None:
#         if not isinstance(thicknesses, dict):
#             # pad it to a list
#             thicknesses = [thicknesses for _ in range(len(result['class_names']))]
#         else:
#             # based on the class name, get the thickness
#             class_names = result['class_names']
#             thicknesses = [thicknesses[class_name] for class_name in class_names]
#         _serialize_thicknesses(thicknesses, rdata)

#     # if text color, label size, or text size are provided
#     if 'text_color' in result or 'label_size' in result or 'text_size' in result:
#         label_property = LabelProperty()

#         if 'text_color' in result:
#             label_property.text_color.r = result['text_color'][0]
#             label_property.text_color.g = result['text_color'][1]
#             label_property.text_color.b = result['text_color'][2]
#             # if alpha is provided, set it
#             if len(result['text_color']) == 4:
#                 label_property.text_color.a = result['text_color'][3]

#         if 'label_size' in result:
#             label_property.label_size = result['label_size']

#         if 'text_size' in result:
#             label_property.text_size = result['text_size']

#         rdata.label_property.CopyFrom(label_property)

#     return rdata.SerializeToString()


# def serialize_face(
def serialize(
        result: List[Any],
        colors: Union[Tuple[int, int, int, int], Dict[str, Tuple[int, int, int, int]], List[Dict[str, Tuple[int, int, int, int]]], None] = None,
        augmentations: Union[Dict[str, str], List[Dict[str, str]], None] = None,
        thicknesses: Union[Dict[str, int], List[Dict[str, int]], None] = None,
        timestamp: int = 0, # timestamp is uint64, use 0 for invalid,
        ) -> str:
    """
    Serialize the recognition result FOR face_landmarks to a protobuf message
    Args:
        result: The recognition result to be serialized 
    Returns:
        The serialized recognition result
    """
    # Create an instance of the top-level RecognitionData message
    rdata = RecognitionData()

    # the fields are all optional, so we need to check if the field exists before serializing it
    _serilizer = {
        'face_landmarks': _serialize_face_landmarks,
        'boxes' : _serialize_face_bounding_boxes,
        # 'face_blendshapes': _serialize_blendshapes,
        # 'position': _serialize_position,
        # 'rotation': _serialize_rotation,
        # colors and augmentations are special cases
    }

    for key, value in result.items():
        if key in _serilizer:
            _serilizer[key](value, rdata)

    # set the timestamp
    rdata.timestamp = timestamp

    # print(f"**************************** result:*******************")
    # print(result)
    # print(f"**************************** colors:*******************")
    # print(colors)
    # print(f"**************************** augmentations:*******************")
    # print(augmentations)
    # print(f"**************************** thicknesses:*******************")
    # print(thicknesses)
    # print(f"**************************** timestamp:*******************")
    # print(timestamp)
    

    # if colors is provided, serialize it
    # if it's a tuple, then it's a single color for all parts of the mesh for all faces and we
    # just need to convert to the proper format: List[List[Tuple[int, int, int, int]]]
    #   (where each tuple is the rgba for ith face and jth face landmark)
    # if it's a dict, then use this color scheme for all recognized faces.
    # if it's a list of dicts, then pass it directly into our serialize function
    if colors is not None:
        if isinstance(colors, tuple):
            # Duplicate mesh colors for each face recognized in the results
            colors_for_each_face = {
                'tesselation_color' : colors,
                'contour_color' : colors,
                'leftBrow_color' : colors,
                'rightBrow_color' : colors,
                'leftEye_color' : colors,
                'rightEye_color' : colors,
                'leftIris_color' : colors,
                'rightIris_color' : colors,
            }
            colors = [colors_for_each_face for _ in range(len(result['face_landmarks']))]
        elif isinstance(colors, dict):
            # If only one color scheme is passed, then use the one color scheme
            # for all faces seen
            # print("*****************Am processing colors as a dict*******************")
            colors = [colors for _ in range(len(result['face_landmarks']))]

        _serialize_colors(colors, rdata)

    # if augmentations is provided, serialize it
    # if it's a single dict of augmentations for each part of the face mesh, then its the same for every face.
    # if it's a list of tuples, then different faces will have their own specified augmentations
    if augmentations is not None:
        if isinstance(augmentations, str):
            # Duplicate mesh augmentations for each face recognized in the results
            augmentations_for_each_face = {
                'tesselation_design' : augmentations,
                'contour_design' : augmentations,
                'leftBrow_design' : augmentations,
                'rightBrow_design' : augmentations,
                'leftEye_design' : augmentations,
                'rightEye_design' : augmentations,
                'leftIris_design' : augmentations,
                'rightIris_design' : augmentations,
            }
            augmentations = [augmentations_for_each_face for _ in range(len(result['face_landmarks']))]
        elif isinstance(augmentations, Dict):
            # print("*****************Am processing augmentations as a dict*******************")
            augmentations = [augmentations for _ in range(len(result['face_landmarks']))]

        _serialize_augmentations(augmentations, rdata)

    # if thicknesses is provided, serialize it
    # if it's a single dict of thicknesses for each part of the face mesh, then its the same for every face.
    # if it's a dict, then it's a thickness for each class
    if thicknesses is not None:
        if isinstance(thicknesses, int):
            # Duplicate mesh augmentations for each face recognized in the results
            thicknesses_for_each_face = {
                'tesselation_thicknesses' : thicknesses,
                'contour_thicknesses' : thicknesses,
                'leftBrow_thicknesses' : thicknesses,
                'rightBrow_thicknesses' : thicknesses,
                'leftEye_thicknesses' : thicknesses,
                'rightEye_thicknesses' : thicknesses,
                'leftIris_thicknesses' : thicknesses,
                'rightIris_thicknesses' : thicknesses,
            }
            thicknesses = [thicknesses_for_each_face for _ in range(len(result['face_landmarks']))]
        elif isinstance(thicknesses, dict):
            # print("*****************Am processing thicknesses as a dict*******************")
            thicknesses = [thicknesses for _ in range(len(result['face_landmarks']))]

        _serialize_thicknesses(thicknesses, rdata)

    # print("rdata before we serialize:")
    # print(rdata)
    return rdata.SerializeToString()


def _serialize_face_landmarks(faces: List[List[Any]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize face landmarks to protobuf message, and add it to the recognition data
    If faces is empty, rdata.faces will remain an empty list
    Args:
        landmarks: The landmarks for each face recognized to be serialized, in the
                   format of [[[x11, y11, z11], ..., [] ...], ...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for face in faces:
        face_landmarks = NormalizedLandmarkList()
        for landmark in face:
            # for every NormalizedLandmark (3d point) in face, serialize
            norm_landmark = NormalizedLandmark()
            norm_landmark.x = landmark.x
            norm_landmark.y = landmark.y
            norm_landmark.z = landmark.z
            norm_landmark.visibility = landmark.visibility
            norm_landmark.presence = landmark.presence
            face_landmarks.landmarks.append(norm_landmark)
        rdata.faces.append(face_landmarks)


def _serialize_face_bounding_boxes(boxes: List[Tuple[int, int, int, int]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize face bounding boxes to protobuf message, and add it to the recognition data
    Args:
        boxes: The bounding boxes for each face, in the format [(x1, y1, x2, y2), (x1, y1, x2, y2),...]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for box in boxes:
        # for every NormalizedLandmark (3d point) in face, serialize
        pbox = Box()
        pbox.x1 = box[0]
        pbox.y1 = box[1]
        pbox.x2 = box[2]
        pbox.y2 = box[3]

        rdata.boxes.append(pbox)


def _serialize_colors(colors: List[Dict[str, Tuple[int, int, int, int]]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize colors to protobuf message, and add it to the recognition data
    Args:
        colors: The face mesh colors to be serialized, in the format of [['tesselation_colors' : ]]
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    # field = kwargs.get('field', 'color')
    # if field == 'color':
    #     to_append = rdata.colors
    # elif field == 'label_color':
    #     to_append = rdata.label_colors
    # else:
    #     raise ValueError(f"Unknown field: {field}")

    for face in colors:
        facemesh_colors = FaceMeshColors()

        # Initialize tesselation colors for this face
        tesselation_color = EdgeColor()
        tesselation_color.r = face['tesselation_color'][0]
        tesselation_color.g = face['tesselation_color'][1]
        tesselation_color.b = face['tesselation_color'][2]
        tesselation_color.a = face['tesselation_color'][3]
        facemesh_colors.faceMesh_tesselation_color.CopyFrom(tesselation_color)

        contour_color = EdgeColor()
        contour_color.r = face['contour_color'][0]
        contour_color.g = face['contour_color'][1]
        contour_color.b = face['contour_color'][2]
        contour_color.a = face['contour_color'][3]
        facemesh_colors.faceMesh_contour_color.CopyFrom(contour_color)

        rightBrow_color = EdgeColor()
        rightBrow_color.r = face['rightBrow_color'][0]
        rightBrow_color.g = face['rightBrow_color'][1]
        rightBrow_color.b = face['rightBrow_color'][2]
        rightBrow_color.a = face['rightBrow_color'][3]
        facemesh_colors.faceMesh_rightBrow_color.CopyFrom(rightBrow_color) 

        leftBrow_color = EdgeColor()
        leftBrow_color.r = face['leftBrow_color'][0]
        leftBrow_color.g = face['leftBrow_color'][1]
        leftBrow_color.b = face['leftBrow_color'][2]
        leftBrow_color.a = face['leftBrow_color'][3]
        facemesh_colors.faceMesh_leftBrow_color.CopyFrom(leftBrow_color) 

        rightEye_color = EdgeColor()
        rightEye_color.r = face['rightEye_color'][0]
        rightEye_color.g = face['rightEye_color'][1]
        rightEye_color.b = face['rightEye_color'][2]
        rightEye_color.a = face['rightEye_color'][3]
        facemesh_colors.faceMesh_rightEye_color.CopyFrom(rightEye_color)

        leftEye_color = EdgeColor()
        leftEye_color.r = face['leftEye_color'][0]
        leftEye_color.g = face['leftEye_color'][1]
        leftEye_color.b = face['leftEye_color'][2]
        leftEye_color.a = face['leftEye_color'][3]
        facemesh_colors.faceMesh_leftEye_color.CopyFrom(leftEye_color)

        rightIris_color = EdgeColor()
        rightIris_color.r = face['rightIris_color'][0]
        rightIris_color.g = face['rightIris_color'][1]
        rightIris_color.b = face['rightIris_color'][2]
        rightIris_color.a = face['rightIris_color'][3]
        facemesh_colors.faceMesh_rightIris_color.CopyFrom(rightIris_color)

        leftIris_color = EdgeColor()
        leftIris_color.r = face['leftIris_color'][0]
        leftIris_color.g = face['leftIris_color'][1]
        leftIris_color.b = face['leftIris_color'][2]
        leftIris_color.a = face['leftIris_color'][3]
        facemesh_colors.faceMesh_leftIris_color.CopyFrom(leftIris_color)

        rdata.faceMesh_colors.append(facemesh_colors)


# def _serialize_mask(masks: List[List[List[int]]], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize masks to protobuf message, and add it to the recognition data
#     Args:
#         mask: The mask to be serialized, in the format of [[[x11, x12, x13], ...], ...]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     Note: the desired mask is a bool mask, but the parameter is a int mask,
#         so we need to convert it to a bool mask
#     """
#     for mask in masks:
#         pmask = Mask()
#         for inner_mask in mask:
#             pinner_mask = InnerMask()
#             pinner_mask.mask.extend([bool(pixel) for pixel in inner_mask])
#             pmask.inner_masks.append(pinner_mask)
#         rdata.masks.append(pmask)


# def _serialize_mask_contour(mask_contour: List[List[Tuple[int, int]]], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize contours to protobuf message, and add it to the recognition data
#     Args:
#         mask_contour: The contour to be serialized, in the format of [(x11, y11), ...]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     for contour in mask_contour:
#         pmask_contour = MaskContour()
#         for point in contour:
#             ppoint = ContourPoint()
#             ppoint.x = point[0]
#             ppoint.y = point[1]
#             pmask_contour.inner_contours.append(ppoint)
#         rdata.mask_contours.append(pmask_contour)


# def _serialize_box(boxes: List[List[int]], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize boxes to protobuf message, and add it to the recognition data
#     Args:
#         box: The box to be serialized, in the format of [x1, y1, x2, y2]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     for box in boxes:
#         pbox = Box()
#         pbox.x1 = box[0]
#         pbox.y1 = box[1]
#         pbox.x2 = box[2]
#         pbox.y2 = box[3]
#         rdata.boxes.append(pbox)


# def _serialize_geometry_center(geometry_center: List[Tuple[int, int]], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize geometry centers to protobuf message, and add it to the recognition data
#     Args:
#         geometry_center: The geometry centers to be serialized, in the format of [(x1, y1), ...]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     for center in geometry_center:
#         pgeometry_center = GeometryCenter()
#         pgeometry_center.x = center[0]
#         pgeometry_center.y = center[1]
#         rdata.geometry_centers.append(pgeometry_center)


# def _serialize_scores(scores: List[float], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize scores to protobuf message, and add it to the recognition data
#     Args:
#         scores: The scores to be serialized
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     rdata.scores.extend(scores)


# def _serialize_class_names(class_names: List[str], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize class names to protobuf message, and add it to the recognition data
#     Args:
#         class_names: The class names to be serialized
#         rdata: The protobuf message to be serialized to
#         kwargs: name_mapping: The mapping from class names to display names
#     Returns:
#         None
#     """
#     name_mapping = kwargs.get('name_mapping', None)
#     if name_mapping is not None:
#         class_names = [name_mapping.get(name, name) for name in class_names]
#     rdata.class_names.extend(class_names)


def _serialize_augmentations(augmentations: List[Dict[str, str]], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize augmentations to protobuf message, and add it to the recognition data
    Args:
        augmentations: The augmentations to be serialized
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for face in augmentations:
        face_augmentation = FaceAugmentations()
        face_augmentation.faceMesh_tesselation_design = face['tesselation_design']
        face_augmentation.faceMesh_contour_design = face['contour_design']
        face_augmentation.faceMesh_leftBrow_design = face['leftBrow_design']
        face_augmentation.faceMesh_rightBrow_design = face['rightBrow_design']
        face_augmentation.faceMesh_leftEye_design = face['leftEye_design']
        face_augmentation.faceMesh_rightEye_design = face['rightEye_design']
        face_augmentation.faceMesh_leftIris_design = face['leftIris_design']
        face_augmentation.faceMesh_rightIris_design = face['rightIris_design']

        rdata.augmentations.append(face_augmentation)


def _serialize_thicknesses(thicknesses: List[float], rdata: RecognitionData, **kwargs) -> None:
    """
    Serialize thicknesses to protobuf message, and add it to the recognition data
    Args:
        thicknesses: The thicknesses to be serialized
        rdata: The protobuf message to be serialized to
    Returns:
        None
    """
    for face in thicknesses:
        contour_thickness = FaceMeshThicknesses()
        contour_thickness.faceMesh_tesselation_thickness = face['tesselation_thickness']
        contour_thickness.faceMesh_contour_thickness = face['contour_thickness']
        contour_thickness.faceMesh_leftBrow_thickness = face['leftBrow_thickness']
        contour_thickness.faceMesh_rightBrow_thickness = face['rightBrow_thickness']
        contour_thickness.faceMesh_leftEye_thickness = face['leftEye_thickness']
        contour_thickness.faceMesh_rightEye_thickness = face['rightEye_thickness']
        contour_thickness.faceMesh_leftIris_thickness = face['leftIris_thickness']
        contour_thickness.faceMesh_rightIris_thickness = face['rightIris_thickness']

        rdata.contour_thicknesses.append(contour_thickness)


# def _serialize_colors(colors: List[List[int]], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize colors to protobuf message, and add it to the recognition data
#     Args:
#         colors: The colors to be serialized, in the format of [[r1, g1, b1], ...]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     field = kwargs.get('field', 'color')
#     if field == 'color':
#         to_append = rdata.colors
#     elif field == 'label_color':
#         to_append = rdata.label_colors
#     else:
#         raise ValueError(f"Unknown field: {field}")

#     for color in colors:
#         pcolor = MaskColor()
#         pcolor.r = color[0]
#         pcolor.g = color[1]
#         pcolor.b = color[2]
#         # if alpha is provided, set it
#         if len(color) == 4:
#             pcolor.a = color[3]
#         to_append.append(pcolor)


# def _serialize_position(position: List[float], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize position to protobuf message, and add it to the recognition data
#     Args:
#         position: The position to be serialized, in the format of [x, y, z]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     pposition = Vec3()
#     pposition.x = position[0]
#     pposition.y = position[1]
#     pposition.z = position[2]
#     rdata.camera_position.CopyFrom(pposition)


# def _serialize_rotation(rotation: List[float], rdata: RecognitionData, **kwargs) -> None:
#     """
#     Serialize rotation to protobuf message, and add it to the recognition data
#     Args:
#         rotation: The rotation to be serialized, in the format of [x, y, z, w]
#         rdata: The protobuf message to be serialized to
#     Returns:
#         None
#     """
#     # protation = Quat()
#     # protation.x = rotation[0]
#     # protation.y = rotation[1]
#     # protation.z = rotation[2]
#     # protation.w = rotation[3]

#     protation = Vec3()
#     protation.x = rotation[0]
#     protation.y = rotation[1]
#     protation.z = rotation[2]
#     rdata.camera_rotation.CopyFrom(protation)