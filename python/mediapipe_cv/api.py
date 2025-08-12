import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Callable, Any, Union, Tuple
from mmdet.structures import DetDataSample
import os
import torch

# mediapipe dependecies
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ben: Change this
config_paths = {
    'face': os.path.join(os.path.dirname(__file__), 'checkpoints/face_landmarker_v2_with_blendshapes.task'),
}
checkpoint_paths = {
    'face': os.path.join(os.path.dirname(__file__), 'checkpoints/face_landmarker_v2_with_blendshapes.task'),
}

device = torch.device('cuda:0')

# build the model from a config file and a checkpoint file
# model = init_detector(config_path, checkpoint_path, device=device)
model = None

# CLASSES = model.dataset_meta['classes']
# COLORS = model.dataset_meta['palette']
CLASSES = None
COLORS = None

executor = ThreadPoolExecutor(max_workers = 30)

def load_model(model_name: str):
    global model
    global CLASSES, COLORS

    config_path = config_paths[model_name]
    checkpoint_path = checkpoint_paths[model_name]

    # Create a FaceLandmarker Object
    base_options = python.BaseOptions(model_asset_path=config_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)

    model = vision.FaceLandmarker.create_from_options(options)

    # CLASSES = model.dataset_meta['classes']
    # COLORS = model.dataset_meta['palette']

# TODO for Ben: Maybe remove?
def get_classes():
    return CLASSES

def get_geometric_center(masks: torch.Tensor) -> List[List[int]]:
    N, H, W = masks.shape

    x = torch.arange(W, device=masks.device).view(1, 1, W).expand(N, H, W)
    y = torch.arange(H, device=masks.device).view(1, H, 1).expand(N, H, W)

    x = (x * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
    y = (y * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))

    return torch.stack([x, y], dim=1).cpu().numpy().astype(int).tolist()

def parse_result(result: DetDataSample,
                 score_threshold: float = 0.15,
                 top_k: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the detection results."""
    pred_instances = result.pred_instances

    # first, filter out the low score boxes
    valid_inds = pred_instances.scores >= score_threshold
    pred_instances = pred_instances[valid_inds]

    # sort the scores
    scores = pred_instances.scores
    _, sort_inds = scores.sort(descending=True)
    pred_instances = pred_instances[sort_inds]

    # get the top_k
    pred_instances = pred_instances[:top_k]

    # get the geometric centers
    centers = get_geometric_center(pred_instances.masks)

    # convert to numpy
    boxes: np.ndarray = pred_instances.bboxes.to(torch.int32).cpu().numpy()
    labels: np.ndarray = pred_instances.labels.cpu().numpy()
    scores: np.ndarray = pred_instances.scores.cpu().numpy()
    masks: np.ndarray = pred_instances.masks.to(torch.uint8).cpu().numpy()

    return masks, boxes, labels, scores, centers


count = 0
total_used_time = 0

# def process_image(image:np.ndarray,
#                   score_threshold: float = 0.3,
#                   top_k: int = 15) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
#     global model
#     global count, total_used_time

#     start_time = time.time()
#     result = model.detect(image)
#     # result = model.detectFromVideo(image, TODO) # TODO
#     used_time = time.time() - start_time
#     total_used_time += used_time
#     count += 1

#     masks, boxes, labels, scores, geometry_center = parse_result(result, score_threshold, top_k)

#     # get contours and geometry centers
#     mask_contours = [None for _ in range(len(masks))]
#     for i, mask in enumerate(masks):
#         # crop the mask by box plus padding of 5 pixels
#         x1, y1, x2, y2 = boxes[i]
#         # x1 = max(0, x1 - 5)
#         # y1 = max(0, y1 - 5)
#         # x2 = min(image.shape[1], x2 + 5)
#         # y2 = min(image.shape[0], y2 + 5)
#         mask_crop = mask[y1:y2, x1:x2]

#         if mask_crop.size == 0:
#             print('mask_crop size is 0', i, CLASSES[labels[i]])
#             print('boxes[i]', boxes[i])
#             print('mask.shape', mask.shape)
#             print('mask_crop.shape', mask_crop.shape)
#             cv2.imwrite('bad_image.png', image)
#         if mask_crop.max() == 0:
#             nonzero = np.nonzero(mask)
#             if len(nonzero) == 0:
#                 print('no nonzero', i, CLASSES[labels[i]])
#             x1 = min(nonzero[1])
#             x2 = max(nonzero[1])
#             y1 = min(nonzero[0])
#             y2 = max(nonzero[0])
#             mask_crop = mask[y1:y2, x1:x2]
#             boxes[i] = [x1, y1, x2, y2]

#         contours = bitmap_to_polygon(mask_crop)

#         # find the largest contour
#         contours.sort(key=lambda x: len(x), reverse=True)
#         largest_contour = max(contours, key = cv2.contourArea)

#         # convert the contour to be relative to the top-left corner of the box
#         # largest_contour = largest_contour + np.array([x1, y1])
#         # largest_contour = largest_contour - np.array([boxes[i][0], boxes[i][1]])

#         mask_contours[i] = largest_contour

#     return masks, mask_contours, boxes, labels, scores, geometry_center

def find_bounding_box_from_landmarks(face_landmarks: List[List[Any]]) -> List[Tuple[int, int, int, int]]:
    """
    Calculate the bounding box for each detected face.
    The boudning box is represented as the NORMALIZED (i.e. between 0 and 1) 2d coordinate for
    the vertex (the top-left most corner) and it's opposite vertex (the bottom-right most corner).
    
    Parameters:
    faces_landmarks (list of list of NormalizedLandmark): 
        A nested list where each inner list contains NormalizedLandmark objects with 'x', 'y', 'z' attributes 
        representing the landmarks for each detected face.
    
    Returns:
    list of dict: A list of dictionaries, each containing the bounding box coordinates for a face.
                  Each dictionary has keys: 'minX', 'maxX', 'minY', 'maxY'.
    """
    # print("***************** PRINT face_landmarks RESULT *****************")
    # print(face_landmarks)
    bounding_boxes = []
    for face in face_landmarks:
        # For every face, loop through its landmarks and create a bounding box from them
        # print("***************** PRINT face *****************")
        # print(face)
        # Extract x and y coordinates for each landmark
        x_coords = [landmark.x for landmark in face]
        y_coords = [landmark.y for landmark in face]
        # print("********************************")
        # print("x_coords: ", x_coords)
        # print("y_coords: ", x_coords)
        
        # Calculate min and max values
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Create a bounding box dictionary for the current face
        bounding_box = (min_x, min_y, max_x, max_y)
        print(f"bounding_box: {bounding_box}")
        bounding_boxes.append(bounding_box)
        
    return bounding_boxes



def get_recognition(image: np.ndarray) -> List[Any]:
    """
        Runs our frame/image through Mediapipe's Face Landmark Model and returns
        a list that contains nested lists of NormalizedLandmarks for each face
        recognized.
    """
    global CLASSES, COLORS

    global model
    global count, total_used_time

    # Need to first convert the image to RGB  (hl2ss decodes into BGR format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image))
    # result = model.detectFromVideo(image, TODO) # TODO
    used_time = time.time() - start_time
    total_used_time += used_time
    count += 1

    # print("***************** PRINT RESULT *****************")
    # print(type(result))
    # print("***************** PRINT face_landmarks RESULT *****************")
    # print(result.face_landmarks)

    # cv2.imshow("main", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.imshow("main", image)

    # print("Got a result from our model!")
    # print(result)

    # Get the bounding box for each detected face. It seems like the face_landmarks model
    # has an intermediate model called Face Detector that outputs a bounding box for each face,
    # but these bounding boxes aren't part of the final output of the FaceLandmarker Mediapipe Model.
    # My naive solution is to just calculate the bounding box from the left-most, right-most, top-most,
    # and bottom-most landmarks. TODO for Ben: Look into optimizing this if it causes performance issues.
    boxes = find_bounding_box_from_landmarks(result.face_landmarks)  # Outputs bounding box for each face: List[(x1, y1, x2, y2), (),...]
    print("Bounding boxes: ", boxes)

    # TODO for 8/5/2025: Try to make sure that the bounding boxes come out right. Draw them!
    annotated_image = draw_face_landmarks_with_boundingBox(rgb_image, result, boxes)
    # print(f"Type of annotated image: {type(annotated_image)}")
    cv2.imshow("main", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)

    # TODO: Once I come back from coffee shop, check to make sure that the bounding boxes are converted into pixel/world space properly on unity side.
    # Also make sure that the bounding boxes are handled properly (min_x, max_y) and (max_x, min_y). I might not doing this properly right now?
    # TODO: Also make sure that I can even just create objects (e.g. a white cube) in the position that the RawImage is "supposed to be". At the very least,
    # a dummy position that should absolutely be in front of the camera (and probably try making it a child of the camera so it's always in front).

    # Ben: I'm going to assume that I can use one of the landmarks in the middle of the face as a "geometric_center"
    # to shoot a ray from to find approximate distance/z-value
    result = {
        "face_landmarks": result.face_landmarks,
        "boxes": boxes,
        # "face_blendshapes": result.face_blendshapes,
        # "masks": masks,
        # "mask_contours": mask_contours,
        # "boxes": boxes,
        # "scores": scores,
        # "labels": labels, # "labels" is the original label, "class_names" is the class name
        # "class_names": class_names,
        # "geometry_center
    }

    return result


async def async_get_recognition(image: np.ndarray,
                                filter_objects: List[str] = [],
                                score_threshold: float = 0.3,
                                top_k: int = 15) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_recognition, image, filter_objects, score_threshold, top_k)


def draw_recognition(image: np.ndarray, result_from_models: List[Tuple[str, Any]]) -> np.ndarray:
    '''
        takes in the image and results from different models and draws the results
        for each.

        Args:
            image (np.ndarray): The image we want to draw our augmentations on
            result_from_models (List[Tuple[str, Any]]): A mapping of the name of the model
                                                 to whatever object stores its results
                                                 (e.g. result_from_models['face_landmarks']
                                                 -> List[List[NormalizedLandmarks]])
    '''

    annotated_image = None

    for model_results in result_from_models:
        task: str = model_results[0]
        results: Any = model_results[1]

        if task == "face_landmarks":
            annotated_image = draw_face_landmarks(image, results)

    return annotated_image


""""""""""""""""HELPER FUNCTIONS FOR DRAWING DIFFERENT AUGMENTATIONS"""""""""""""""

def draw_face_landmarks(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(  # solutions is a mediapipe lib imported in
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image


def draw_face_landmarks_with_boundingBox(rgb_image, detection_result, bounding_boxes):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    bounding_box = bounding_boxes[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(  # solutions is a mediapipe lib imported in
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())
    

    #Draw the bounding box around each detected face
    # print(f"(bounding_box[0], bounding_box[2]): ({bounding_box[0], bounding_box[2]})")
    print(f"^^^^^^^^ shape of rgb_image: {rgb_image.shape}")
    image_rows, image_cols, _ = rgb_image.shape
    pt1 = solutions.drawing_utils._normalized_to_pixel_coordinates(bounding_box[0], bounding_box[3], image_cols, image_rows)  # (max_x, min_y)
    pt2 = solutions.drawing_utils._normalized_to_pixel_coordinates(bounding_box[2], bounding_box[1], image_cols, image_rows)  # (min_x, max_y)
    print(f"&&&&&& pt1: {pt1} | pt2: {pt2}")
    cv2.rectangle(annotated_image, pt1, pt2, color=(255, 0, 0), thickness=10)

  return annotated_image

"""""""""""""""""""""""""" """"""""""""""""""""""""""

def get_filtered_objects(result: Dict[str, Any],
                         filter_objects: List[str] = [],
                         exclude_objects: List[str] = []) -> Dict[str, Any]:
    """Filter the objects by class names."""
    assert 'class_names' in result

    fields = result.keys()
    new_result = {field: [] for field in fields}
    class_names = result['class_names']

    for i, class_name in enumerate(class_names):
        if (not filter_objects or class_name in filter_objects) and class_name not in exclude_objects:
            for field in fields:
                new_result[field].append(result[field][i])

    return new_result


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    outs = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[-2]
    contours = [c.reshape(-1, 2) for c in contours]
    return contours


def main():
    capture = cv2.VideoCapture(0)
    # set resolution to 640x360
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("Resolution:", capture.get(cv2.CAP_PROP_FRAME_WIDTH), "x", capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    total_used_time = 0

    while True:
        ret, image = capture.read()
        if not ret:
            break

        # plot image
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        start_time = time.time()
        result = get_recognition(image, score_threshold=0.5)
        used_time = time.time() - start_time
        total_used_time += used_time
        count += 1

        if count > 30:
            latency = total_used_time / count
            print(f'latency: {latency:.3f}s ({1/latency:.3f} fps)')
            count = 0
            total_used_time = 0

    # image = async_draw_recognition(image, result, draw_contour=True, draw_text=True, draw_score=True)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)

    # bad_image = cv2.imread('bad_image.png')
    # print(bad_image.shape, bad_image.dtype)

    # result = inference_detector(model, bad_image)
    # ins = result.pred_instances
    # ins = ins[ins.scores >= 0.3]
    # boxes: np.ndarray = ins.bboxes.to(torch.int32).cpu().numpy()
    # labels: np.ndarray = ins.labels.cpu().numpy()
    # scores: np.ndarray = ins.scores.cpu().numpy()
    # masks: np.ndarray = ins.masks.to(torch.uint8).cpu().numpy()

    # for i, mask in enumerate(masks):
    #     print(i, CLASSES[labels[i]], mask.shape, boxes[i], scores[i])

if __name__ == '__main__':
    main()