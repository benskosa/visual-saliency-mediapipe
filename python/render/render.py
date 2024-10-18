import cv2
import numpy as np
import ujson as json
import argparse
from typing import Tuple, List, Union, Dict, Any
import imageio

# colors in BGR
builtin_colors = {
    'yellow': (0, 255, 255),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'darkyellow': (14, 188, 188),
    'orange': (0, 165, 255),
}

other_color = {
    'yellow': 'black',
}

def lerp_lab(color1: Tuple, color2: Tuple, t: float) -> Tuple:
    lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2Lab).astype(float)
    lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2Lab).astype(float)
    interpolated_lab = t * lab1 + (1-t) * lab2
    interpolated_rgb = cv2.cvtColor(interpolated_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return tuple(interpolated_rgb[0][0].tolist())


def draw_contour(image, objs, color, thickness = 15):
    print('drawing contour with color', color)
    for obj in objs:
        # draw contour
        cv2.drawContours(image, [np.array(obj['contour'])], -1, color, thickness)
    return image


def draw_solid(image, objs, color, alpha=0.75):
    print('drawing solid with color', color)
    # since drawPoly doesn't support alpha, we need to crop out the region first
    # first, find the bounding box
    for obj in objs:
        x, y, w, h = cv2.boundingRect(np.array(obj['contour']))
        # crop the region
        roi = image[y:y+h, x:x+w].astype(float)
        # create a mask
        filled_mask = np.zeros_like(roi, dtype=float)
        filled_mask[:,:,0] = color[0]
        filled_mask[:,:,1] = color[1]
        filled_mask[:,:,2] = color[2]
        # create a 0-1 mask
        mask = np.zeros_like(roi, dtype=float)
        cv2.fillPoly(mask, [np.array(obj['contour']) - [x, y]], (1, 1, 1))
        # blend the image
        res = (1-mask)*roi + (filled_mask * alpha + (1-alpha)*roi) * mask
        image[y:y+h, x:x+w] = res.astype(np.uint8)
    return image


def work(name, color='yellow', mode='outline', thickness=15):
    image = cv2.imread(name)
    json_file = name.split('.')[0] + '.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    objects = data

    thickness = round(thickness / 1280 * min(image.shape[:2]))

    # if mode starts with "diff", we will filter out important/not important objects
    if mode.startswith('diff'):
        important_objs = [obj for obj in data if obj['important']]
        not_important_objs = [obj for obj in data if not obj['important']]

        # first draw the not important objects
        if mode.endswith('design'):
            # for design, use outline for not important objects, use solid for important objects
            image = draw_contour(image, not_important_objs, builtin_colors[color], thickness)
            mode = 'solid'
        elif mode.endswith('color'):
            # for color, use the other color for not important objects
            image = draw_contour(image, not_important_objs, builtin_colors[other_color[color]], thickness)
            mode = 'outline'
        elif mode.endswith('flashing'):
            # just draw the flashing for important objects, use static outline for not important objects
            image = draw_contour(image, not_important_objs, builtin_colors[color], thickness)
            mode = 'flashing'
        
        objects = important_objs


    if mode == 'outline':
        image_copy = image.copy()
        image_copy = draw_contour(image_copy, objects, builtin_colors[color], thickness)
        cv2.imwrite(name.split('.')[0] + '_outline.jpg', image_copy)

    elif mode == 'solid':
        image_copy = image.copy()
        image_copy = draw_solid(image_copy, objects, builtin_colors[color])
        cv2.imwrite(name.split('.')[0] + '_solid.jpg', image_copy)

    elif mode == 'flashing':
        frames = []
        steps = 7
        for i in range(-steps, steps+1):
            progress = abs(i) / steps
            prog_color = lerp_lab(builtin_colors[color], builtin_colors['black'], progress)
            image_copy = image.copy()
            image_copy = draw_contour(image_copy, objects, prog_color, thickness)
            image2 = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            # scale to 0.5 to reduce file size
            image2 = cv2.resize(image2, (image2.shape[1]//2, image2.shape[0]//2))
            frames.append(image2)
        imageio.mimsave(name.split('.')[0] + '_flashing.gif', frames, format='GIF', duration=0.02, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='kitchen2.png', help='name of the image')
    parser.add_argument('--color', type=str, default='yellow', help='yellow/red/blue/green/black/white/cyan/magenta/darkyellow')
    parser.add_argument('--thickness', type=int, default=12, help='thickness of the outline' )
    parser.add_argument('--mode', type=str, default='outline',
                        help='solid/outline/flashing/diff-design/diff-color/diff-flashing')

    args = parser.parse_args()
    work(args.name, args.color, args.mode, args.thickness)