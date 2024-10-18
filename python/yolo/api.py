import os
import cv2
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Colors
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import imageio
from pygifsicle import optimize

from typing import List, Dict, Callable, Any, Union, Tuple
import torch
import numpy as np
import time
from tqdm import trange, tqdm

import traceback
import ujson as json


model_path = os.path.join(os.path.dirname(__file__), 'pt', 'yolov8x-seg.pt')

# model_path = os.path.join(os.path.dirname(__file__), 'pt', 'best-outdoor.pt')
# model_path = os.path.join(os.path.dirname(__file__), 'pt', 'best-freeze.pt')
# model_path = os.path.join(os.path.dirname(__file__), 'pt', 'best-ps-mapillary-tuned.pt')
# config_json = os.path.join(os.path.dirname(__file__), 'pt', 'config_v2.0.json')
# with open(config_json, 'r', encoding='utf-8') as f:
#     METADATA = json.load(f)
#     LABELS = METADATA['labels']
#     class_mapping = {label['name']: label['readable'] for label in LABELS}


# model_path = os.path.join(os.path.dirname(__file__), 'pt', 'best-kitchen.pt')

model = YOLO(model_path)

CLASSES: Dict[int, str] = model.names
REV_CLASSES: Dict[str, int] = {name: i for (i, name) in CLASSES.items()}
COLORS = Colors()

_builtin_colors = {
    'yellow': (0, 255, 255),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
}

executor = ThreadPoolExecutor(max_workers = 30)

def get_geometric_center(masks: torch.Tensor) -> np.ndarray:
    N, H, W = masks.shape

    x = torch.arange(W, device=masks.device).view(1, 1, W).expand(N, H, W)
    y = torch.arange(H, device=masks.device).view(1, H, 1).expand(N, H, W)

    x = (x * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
    y = (y * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))

    return torch.stack([x, y], dim=1).to(torch.int32).cpu().numpy()

def parse_result(result: "Results", width, height, dwidth, dheight) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the detection results."""
    # score threshold is already handled by the model
    # top_k is handled by the model

    # if no object is detected, return empty np arrays
    if len(result) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # get the masks and cast to uint8
    masks_torch = result.masks.data.to(torch.uint8)
    # crop the masks to the image size
    # masks_torch = masks_torch[:, dheight//2-height//2:dheight//2+height//2, dwidth//2-width//2:dwidth//2+width//2]

    # get the geometric centers
    centers = get_geometric_center(masks_torch)

    # convert to numpy
    boxes: np.ndarray = result.boxes.xyxy.to(torch.int32).cpu().numpy()
    labels: np.ndarray = result.boxes.cls.to(torch.int32).cpu().numpy()
    scores: np.ndarray = result.boxes.conf.cpu().numpy()
    masks: np.ndarray = masks_torch.cpu().numpy()

    return masks, boxes, labels, scores, centers


def process_image(image:np.ndarray,
                  score_threshold: float = 0.3,
                  top_k: int = 15,
                  filter_classes: Union[List[int], None] = None) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    global model

    # compute the image size
    height, width, _ = image.shape
    # get the smallest multiple of 32 that is larger than the image size
    dwidth = width + 32 - width % 32
    dheight = height + 32 - height % 32

    results = model.predict(image, imgsz=(dheight, dwidth), conf=score_threshold,
    # results = model.predict(image, imgsz=640, conf=score_threshold,
                    max_det=top_k, device='cuda:0', verbose=False, half=True,
                    classes=filter_classes, retina_masks=True)
    result = results[0]

    masks, boxes, labels, scores, geometry_center = parse_result(result, width, height, dwidth, dheight)

    # get contours and geometry centers
    mask_contours = [None for _ in range(len(masks))]
    for i, mask in enumerate(masks):
        # crop the mask by box plus padding of 5 pixels
        x1, y1, x2, y2 = boxes[i]
        # x1 = max(0, x1 - 5)
        # y1 = max(0, y1 - 5)
        # x2 = min(image.shape[1], x2 + 5)
        # y2 = min(image.shape[0], y2 + 5)
        mask_crop = mask[y1:y2, x1:x2]

        if mask_crop.max() == 0:
            nonzero = np.nonzero(mask)
            if len(nonzero[0]) == 0 or len(nonzero[1]) == 0:
                print('no nonzero', i, CLASSES[labels[i]])
            x1 = min(nonzero[1])
            x2 = max(nonzero[1])
            y1 = min(nonzero[0])
            y2 = max(nonzero[0])
            mask_crop = mask[y1:y2, x1:x2]
            boxes[i] = [x1, y1, x2, y2]

        contours = bitmap_to_polygon(mask_crop)
        # find the largest contour
        contours.sort(key=lambda x: len(x), reverse=True)
        largest_contour = max(contours, key = cv2.contourArea)

        mask_contours[i] = largest_contour

    return masks, mask_contours, boxes, labels, scores, geometry_center


def get_recognition(image: np.ndarray,
                    filter_objects: List[str] = [],
                    score_threshold: float = 0.3,
                    top_k: int = 15,
                    include_classes: List[int] = [],
                    class_mapping: Dict[str, str] = None,
                    color: Union[str, Tuple[int, int, int], None] = None) -> Dict[str, Any]:
    global CLASSES, COLORS

    if filter_objects and include_classes:
        raise ValueError("Cannot specify both filter_objects and include_classes.")
    if filter_objects:
        object_ids = [REV_CLASSES[object_name] for object_name in filter_objects if object_name in REV_CLASSES]
    elif include_classes:
        object_ids = include_classes
    else:
        object_ids = None

    masks, mask_contours, boxes, labels, scores, geometry_center = process_image(
        image,
        score_threshold,
        top_k,
        object_ids,
    )

    mask_contours = [contour.tolist() for contour in mask_contours]
    geometry_center = geometry_center.tolist()
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()

    # convert labels to class names
    class_names = [CLASSES[label] for label in labels]
    if class_mapping:
        class_names = [class_mapping.get(class_name, class_name) for class_name in class_names]

    # get colors
    # each color is a 3-tuple (R, G, B)
    # bgr2rgb = lambda x: (x[2], x[1], x[0])
    # if color:
    #     if isinstance(color, tuple):
    #         color_list = [bgr2rgb(color) for _ in labels]
    #     elif color in _builtin_colors:
    #         # builtin colors are in BGR, convert to RGB
    #         color_list = [bgr2rgb(_builtin_colors[color]) for _ in labels]
    #     else:
    #         raise ValueError(f"Unknown color {color}")
    # else:
    #     color_list = [COLORS(label, False) for label in labels]

    result = {
        "masks": masks,
        "mask_contours": mask_contours,
        "boxes": boxes,
        "scores": scores,
        "labels": labels, # "labels" is the original label, "class_names" is the class name
        "class_names": class_names,
        "geometry_center": geometry_center,
        # "colors": color_list,
    }

    return result


async def async_get_recognition(image: np.ndarray,
                                filter_objects: List[str] = [],
                                score_threshold: float = 0.3,
                                top_k: int = 15) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_recognition, image, filter_objects, score_threshold, top_k)


def draw_recognition(image: np.ndarray, result: Dict[str, Any],
                     black: bool = False, draw_contour: bool = False, draw_mask: bool = True, 
                     draw_box: bool = False, draw_text: bool = True, draw_score = True,
                     draw_center = False, increase_contrast = False,
                     lv_color: Tuple[int, int, int] = None,
                     contour_thickness: int = 15,
                     alpha: float = 0.45) -> np.ndarray:
    masks = result['masks']
    mask_contours = result['mask_contours']
    boxes = result['boxes']
    class_names = result['class_names']
    scores = result['scores']
    geometry_center = result['geometry_center']
    labels = result['labels']
    
    if black:
        image = np.zeros_like(image)
    else:
        # else, make a copy
        image = image.copy()

    if len(masks) == 0:
        return image
    
    has_alpha = image.shape[2] == 4

    # colors
    # each color is a 3-tuple (B, G, R) or (B, G, R, A)
    colors = []
    color_list = []
    for label in labels:
        if lv_color is not None:
            color = lv_color
            # if has alpha, add the alpha channel
            if has_alpha and len(color) == 3:
                color = (*color, 255)
        else:
            color = COLORS(label, True)
            color = (255, color[2], color[1], color[0]) if has_alpha else (color[2], color[1], color[0])
        color_list.append(color)
        colors.append(np.array(color, dtype=float).reshape(1,1,1,-1))
    colors = np.concatenate(colors, axis=0)

    # yield to other tasks
    # await asyncio.sleep(0)
    
    if draw_mask:
        # masks N*H*W
        masks = np.array(masks, dtype=float)
        # change to N*H*W*1
        masks = np.expand_dims(masks, axis=3)

        masks_color = masks.repeat(3, axis=3) * colors * alpha

        inv_alpha_masks = masks * (-alpha) + 1

        masks_color_summand = masks_color[0]
        if len(masks_color) > 1:
            inv_alpha_cumul = inv_alpha_masks[:-1].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alpha_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        image = image * inv_alpha_masks.prod(axis=0) + masks_color_summand
        image = image.astype(np.uint8)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw the contours
    if draw_contour:
        for i, contour in enumerate(mask_contours):
            # contour is relative to the box, need to add the box's top-left corner
            x1, y1, _, _ = boxes[i]
            contour = np.array(contour) + np.array([x1, y1])
            contour = np.array(contour, dtype=np.int32)
            color = color_list[i]
            cv2.drawContours(image, [contour], -1, color, thickness = contour_thickness)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw box
    if draw_box:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = color_list[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, contour_thickness)

    # yield to other tasks
    # await asyncio.sleep(0)

    # place text at the center
    if draw_text:
        for i, center in enumerate(geometry_center):
            text = class_names[i]
            if draw_score:
                text += f' {scores[i]:.2f}'
            
            fontsize = 1
            fontthickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            textsize = cv2.getTextSize(text, font, fontsize, fontthickness)[0]
            textX = center[0] - textsize[0] // 2
            textY = center[1] + textsize[1] // 2

            cv2.putText(image, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), fontthickness)

    # draw center with a green circle, and center of bounding box with a yellow circle
    if draw_center:
        for i, center in enumerate(geometry_center):
            cv2.circle(image, tuple(center), 3, (0, 255, 0), -1)
            x1, y1, x2, y2 = boxes[i]
            center_box = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(image, center_box, 3, (0, 255, 255), -1)


    if increase_contrast:
        contrast_factor = 1.2
        dim_factor = 0.8
        brightness_factor = 0

        # masks N*H*W
        masks = np.array(masks, dtype=np.uint8)
        # all_masks: H*W
        all_masks = np.clip(np.sum(masks, axis=0), 0, 1)
        # all_masks: H*W*3
        all_masks = np.stack([all_masks, all_masks, all_masks], axis=2).astype(np.uint8)

        # only increase contrast and brightness of interested area
        image_with = image * all_masks
        image_without = image * (1 - all_masks)
        image_with_enhanced = cv2.convertScaleAbs(image_with, alpha=contrast_factor, beta=brightness_factor)
        image_without_dim = cv2.convertScaleAbs(image_without, alpha=dim_factor, beta=brightness_factor)
        image = image_with_enhanced + image_without_dim

    
    return image


def get_filtered_objects(result: Dict[str, Any], filter_objects: List[str]) -> Dict[str, Any]:
    """Filter the objects by class names."""
    assert 'class_names' in result

    fields = result.keys()
    new_result = {field: [] for field in fields}
    class_names = result['class_names']

    for i, class_name in enumerate(class_names):
        if class_name in filter_objects:
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


def lerp_lab(color1: Tuple, color2: Tuple, t: float) -> Tuple:
    lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2Lab).astype(float)
    lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2Lab).astype(float)
    interpolated_lab = t * lab1 + (1-t) * lab2
    interpolated_rgb = cv2.cvtColor(interpolated_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return tuple(interpolated_rgb[0][0].tolist())


def export_json(results: Dict[str, Any], importance: List[bool], direction: List[str], filename: str):
    # in the json: a list of object attributes
    # 1. contour: list of points
    # 2. center: (x, y)
    # 3. important: true or false
    # 4. class: class name
    # 5. direction: the direction of the text label ("up" or "down")

    contours = results['mask_contours']
    centers = results['geometry_center']
    class_names = results['class_names']

    output = []
    for i, (contour, center, important, class_name, dir) in enumerate(zip(contours, centers, importance, class_names, direction)):
        output.append({
            'contour': contour,
            'center': center,
            'important': important,
            'class': class_name,
            'direction': dir,
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f)

def main():
    pass
    image_folder = 'dummy-pic'
    image_files = [s for s in os.listdir(image_folder) if s.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        result = get_recognition(image, score_threshold=0.45, top_k=50)

        wanted_fields = [
            # 'masks',
            'mask_contours',
            'boxes',
            # 'labels',
            'scores',
            'class_names',
            'geometry_center',
        ]
        result_filtered = {field: result[field] for field in wanted_fields}

        # export to JSON
        fname = image_file.replace('.png', '.json')
        json_path = os.path.join(image_folder, fname)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_filtered, f)


if __name__ == '__main__':
    main()