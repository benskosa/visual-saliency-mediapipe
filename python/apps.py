#------------------------------------------------------------------------------
# This script receives encoded video from the HoloLens cameras and plays it.
# Press esc to stop.
#------------------------------------------------------------------------------
import multiprocessing as mp
import threading
import numpy as np
import cv2
import os
import hl2ss
from hl2ss import _packet, umq_command_buffer
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import time
import asyncio
import argparse
from utils.serializer import serialize
from utils.utils import self_ip
from modelprocess import ToyProcess

# from yolo.api import draw_recognition, get_recognition
# included_classes = [
#     'bowl',
#     'spoon',
#     'fork',
#     'knife',
#     'scissors',
#     'cup',
#     'bottle',
#     'wine glass',
#     'ladle',
#     'spatula',
# ]

# from mmdetection.api import get_recognition, draw_recognition, load_model, get_classes
from mediapipe.api import get_recognition, draw_recognition, load_model, get_classes
# scenario = 'kitchen'
# scenario = 'vistas'
scenario = 'face'

exclude_classes = ['object--vehicle--bus', 'object--vehicle--car', 'object--vehicle--truck', 'nature--vegetation', 'object--traffic-sign--front', 'carafe']
included_classes = []


from typing import Tuple, Dict, Any, Union, List, Callable
import traceback
from collections import deque # for storing previous results

try:
    import ujson as json
except ImportError:
    print("ujson not found, using json")
    import json

# UI --------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Combobox

tk_root = None
color_lock = threading.Lock()
design_lock = threading.Lock()
contour_lock = threading.Lock()
label_lock = threading.Lock()
metadata_lock = threading.Lock()
pause_lock = threading.Lock()

# Global method: onclose ------------------------------------------------------
g_on_close = None

# Global variables ------------------------------------------------------------
g_values = {
    'primary_color': 'yellow',
    'secondary_color': 'blue',
    'primary_alpha': 80,
    'secondary_alpha': 80,
    'primary_label_color': 'yellow',
    'secondary_label_color': 'blue',
    'primary_label_alpha': 80,
    'secondary_label_alpha': 80,
    'primary_design': 'outline',
    'secondary_design': 'outline',
    'primary_contour_thickness': 3,
    'secondary_contour_thickness': 3,
}
# RGB color mapping
g_color_mapping = {
    'yellow': (255, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
}
g_color_options = list(g_color_mapping.keys())
g_alpha_options = [5] + list(range(10, 101, 10)) # 10 - 100, step 10

g_design_options = [
    'outline',
    'solid',
    # 'flash-solid',
    # 'text',
    # 'text-arrow',
    # 'flash-outline',
    # 'outline+text-arrow',
    # 'icon-arrow',
    'icon-arrow-nobg',
    # 'outline+icon-arrow',
    'outline+icon-arrow-nobg',
]

g_text_alpha = 100
g_label_size = 1 # 0.5 - 2.5, step 0.1
g_label_size_options: List[float] = list(range(5, 26))
g_text_size = 36 # 12 - 72, step 6
g_text_size_options: List[int] = np.arange(12, 78, 6).tolist()

g_contour_thickness_options: List[int] = np.arange(1, 9, 1).tolist()

g_metadata: Dict[str, Dict[str, Any]] = {}
g_use_diff_vis = False
g_name_mapping = {}
g_diff_colors = {}
g_diff_thicknesses = {}
g_diff_label_colors = {}
g_diff_designs = {}
g_current_metadata = 'all'

g_pause = False

def get_metadata(name) -> Dict[str, Any]:
    global g_metadata
    metadata_path = os.path.join('metadata', f"{name}.json")
    with metadata_lock:
        with open(metadata_path, 'r', encoding = 'utf-8') as f:
            g_metadata = json.load(f)

def update_diffs():
    global g_values, g_color_mapping, g_use_diff_vis, g_diff_colors, g_diff_designs, g_diff_thicknesses
    global g_diff_label_colors, g_metadata, g_name_mapping

    with color_lock:
        primary_color = (*g_color_mapping[g_values['primary_color']], g_values['primary_alpha'])
        secondary_color = (*g_color_mapping[g_values['secondary_color']], g_values['secondary_alpha'])
        primary_label_color = (*g_color_mapping[g_values['primary_label_color']], g_values['primary_label_alpha'])
        secondary_label_color = (*g_color_mapping[g_values['secondary_label_color']], g_values['secondary_label_alpha'])

    with design_lock:
        primary_design = g_values['primary_design']
        secondary_design = g_values['secondary_design']

    with contour_lock:
        primary_contour_thickness = g_values['primary_contour_thickness']
        secondary_contour_thickness = g_values['secondary_contour_thickness']

    with metadata_lock:
        g_diff_colors = {
            class_name: primary_color if prop['priority'] == 'high' else secondary_color
            for class_name, prop in g_metadata.items()
        }
        g_diff_designs = {
            class_name: primary_design if prop['priority'] == 'high' else secondary_design
            for class_name, prop in g_metadata.items()
        }
        g_diff_thicknesses = {
            class_name: primary_contour_thickness if prop['priority'] == 'high' else secondary_contour_thickness
            for class_name, prop in g_metadata.items()
        }
        g_diff_label_colors = {
            class_name: primary_label_color if prop['priority'] == 'high' else secondary_label_color
            for class_name, prop in g_metadata.items()
        }
        g_name_mapping = {
            class_name: prop.get('display_name', class_name)
            for class_name, prop in g_metadata.items()
        }

get_metadata(g_current_metadata)
update_diffs()

# Settings --------------------------------------------------------------------
VIDEO_PORT = hl2ss.StreamPort.PERSONAL_VIDEO
VIDEO_UDP_PORT = hl2ss.StreamUDPPort.PERSONAL_VIDEO

# PV parameters
pv_width     = 640
pv_height    = 360
pv_framerate = 30

# Maximum number of frames in buffer
buffer_elements = 150

# printing threshold
print_threshold = 5

# bandwidth: 2.3MB/s
data_bandwidth = 2.3 * 1024 * 1024

# Command buffer to send data -------------------------------------------------
class command_buffer(hl2ss.umq_command_buffer):
    # Command structure
    # id:     u32 (4 bytes)
    # size:   u32 (4 bytes)
    # params: size bytes

    def debug_message(self, text: bytes, command_type: int = 0xFFFFFFFE) -> None:
        # Command id: 0xFFFFFFFE
        # Command params: string encoded as utf-8
        self.add(command_type, text)

# ----------------------------------------------------------------------------
last_print_time_bytes = time.time()
count_bytes = 0
bandwidth: float = 0

def run_ui() -> None:
    global tk_root

    def update_color(prefix: str, combobox: Combobox) -> None:
        global g_values
        val = combobox.current()
        with color_lock:
            g_values[prefix + '_color'] = g_color_options[int(val)]
        update_diffs()
        print(f"{prefix.capitalize()} color changed to {g_values[prefix + '_color']}")

    def update_alpha(prefix: str, combobox: Combobox) -> None:
        global g_values
        val = combobox.current()
        with color_lock:
            g_values[prefix + '_alpha'] = g_alpha_options[int(val)]
        update_diffs()
        print(f"{prefix.capitalize()} alpha changed to {g_values[prefix + '_alpha']}")

    def update_design(prefix: str, combobox: Combobox) -> None:
        global g_values
        val = combobox.current()
        with design_lock:
            g_values[prefix + '_design'] = g_design_options[int(val)]
        update_diffs()
        print(f"{prefix.capitalize()} design changed to {g_values[prefix + '_design']}")

    def update_label_size(event) -> None:
        global g_label_size
        value = labelsize_combobox.current()
        with label_lock:
            g_label_size = g_label_size_options[int(value)] / 10
            print(f"Label size changed to {g_label_size}")

    def update_contour_thickness(prefix: str, combobox: Combobox) -> None:
        global g_values
        value = combobox.current()
        with contour_lock:
            g_values[prefix + '_contour_thickness'] = g_contour_thickness_options[int(value)]
        update_diffs()
        print(f"{prefix.capitalize()} contour thickness changed to {g_values[prefix + '_contour_thickness']}")

    def on_diff_checbox_toggle() -> None:
        global g_use_diff_vis
        with metadata_lock:
            g_use_diff_vis = diff_checkbox_var.get()
        print(f"Differentiating visualizations {'enabled' if g_use_diff_vis else 'disabled'}")

    def on_pause() -> None:
        global g_pause
        with pause_lock:
            g_pause = not g_pause
        if g_pause:
            pause_button.config(text="RESUME", bg='green', fg='white')
        else:
            pause_button.config(text="PAUSE", bg='orange', fg='black')

        print(f"{'Paused' if g_pause else 'Resumed'} the program")

    def on_stop() -> None:
        global g_on_close
        if g_on_close is not None:
            g_on_close()
        print("Stopping the program")


    tk_root = tk.Tk()
    tk_root.title("Control Panel")

    style = ttk.Style()
    # adjust the drop down list's width
    style.layout('TCombobox', [
        ('Combobox.border', {
            'children': [
                ('Combobox.padding', {
                    'children': [
                        ('Combobox.focus', {
                            'children': [
                                ('Combobox.textarea', {'sticky': 'nswe'})
                            ]
                        })
                    ]
                })
            ],
            'sticky': 'nswe'
        })
    ])

    style.configure('TCombobox', arrowsize=20, padding=10)

    current_row = 0

    for i, prefix in enumerate(('primary', 'secondary')):
        color_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Color:")
        color_label.grid(row=current_row, column=0, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        color_combobox = ttk.Combobox(tk_root, values=g_color_options, style='TCombobox', state='readonly')
        color_combobox.current(g_color_options.index(g_values[prefix + '_color']))
        color_combobox.grid(row=current_row + 1, column=0, padx=10, pady=5)
        color_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=color_combobox: update_color(p, c))

        alpha_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Alpha:")
        alpha_label.grid(row=current_row, column=1, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        alpha_combobox = ttk.Combobox(tk_root, values=g_alpha_options, style='TCombobox', state='readonly')
        alpha_combobox.current(g_alpha_options.index(g_values[prefix + '_alpha']))
        alpha_combobox.grid(row=current_row + 1, column=1, padx=10, pady=5)
        alpha_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=alpha_combobox: update_alpha(p, c))

        current_row += 2

    for i, prefix in enumerate(('primary', 'secondary')):
        color_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Label Color:")
        color_label.grid(row=current_row, column=0, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        color_combobox = ttk.Combobox(tk_root, values=g_color_options, style='TCombobox', state='readonly')
        color_combobox.current(g_color_options.index(g_values[prefix + '_label_color']))
        color_combobox.grid(row=current_row + 1, column=0, padx=10, pady=5)
        color_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=color_combobox: update_color(p + '_label', c))

        alpha_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Label Alpha:")
        alpha_label.grid(row=current_row, column=1, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        alpha_combobox = ttk.Combobox(tk_root, values=g_alpha_options, style='TCombobox', state='readonly')
        alpha_combobox.current(g_alpha_options.index(g_values[prefix + '_label_alpha']))
        alpha_combobox.grid(row=current_row + 1, column=1, padx=10, pady=5)
        alpha_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=alpha_combobox: update_alpha(p + '_label', c))

        current_row += 2

    for i, prefix in enumerate(('primary', 'secondary')):
        thickness_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Contour Thickness:")
        thickness_label.grid(row=current_row, column=i, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        thickness_combobox = ttk.Combobox(tk_root, values=g_contour_thickness_options, style='TCombobox', state='readonly')
        thickness_combobox.current(g_contour_thickness_options.index(g_values[prefix + '_contour_thickness']))
        thickness_combobox.grid(row=current_row + 1, column=i, padx=10, pady=5)
        thickness_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=thickness_combobox: update_contour_thickness(p, c))

    current_row += 2

    # label size label
    label_size_label = tk.Label(tk_root, text="Label Size:")
    label_size_label.grid(row=current_row, column=0, columnspan=2, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

    # label size
    labelsize_combobox = ttk.Combobox(tk_root, values=g_label_size_options, style='TCombobox', state='readonly')
    labelsize_combobox.current(g_label_size_options.index(g_label_size*10))
    labelsize_combobox.grid(row=current_row+1, column=0, columnspan=2, padx=10, pady=5)
    labelsize_combobox.bind("<<ComboboxSelected>>", update_label_size)

    current_row += 2

    for i, prefix in enumerate(('primary', 'secondary')):
        design_label = tk.Label(tk_root, text=f"{prefix.capitalize()} Design:")
        design_label.grid(row=current_row, column=i, padx=10, pady=(5, 0))  # Padding on top, but none on the bottom

        design_combobox = ttk.Combobox(tk_root, values=g_design_options, style='TCombobox', state='readonly')
        design_combobox.current(g_design_options.index(g_values[prefix + '_design']))
        design_combobox.grid(row=current_row + 1, column=i, padx=10, pady=5)
        design_combobox.bind("<<ComboboxSelected>>", lambda event, p=prefix, c=design_combobox: update_design(p, c))

    current_row += 2

    # diff checkbox
    diff_checkbox_var = tk.BooleanVar()
    diff_checkbox = tk.Checkbutton(tk_root, text="Differentiate Visualizations", variable=diff_checkbox_var, command=on_diff_checbox_toggle)
    diff_checkbox.grid(row=current_row, column=0, columnspan=2, padx=10, pady=5)
    diff_checkbox_var.set(g_use_diff_vis)

    current_row += 1

    # pause button
    pause_button = tk.Button(tk_root, text="PAUSE", command=on_pause, bg='orange', fg='black', font=('Arial', 16))
    pause_button.grid(row=current_row, column=0, padx=10, pady=5)

    # stop button
    stop_button = tk.Button(tk_root, text="STOP", command=on_stop, bg='red', fg='white', font=('Arial', 16))
    stop_button.grid(row=current_row, column=1, padx=10, pady=5)

    current_row += 1

    tk_root.mainloop()



def recv_data_callback(data: bytes) -> None:
    global last_print_time_bytes, count_bytes, bandwidth
    count_bytes += len(data)
    timediff = time.time() - last_print_time_bytes
    if count_bytes > 0 and timediff > 1:
        # display in Kbps, should multiply by 8 and divide by 1000
        bandwidth = count_bytes * 8 / 1000 / timediff
        count_bytes = 0
        last_print_time_bytes = time.time()


class VideoProcessApp:
    def __init__(self):
        self.latest_frame: np.ndarray = None
        self.latest_pose: List[List[float]] = None
        self.latest_timestamp: int = 0
        # a result is a tuple of (frame, result, pose, timestamp)
        self.latest_result: Tuple[np.ndarray, Dict[str, Any], Union[List[List[float]], None], int] = None

        self.count_recv = 0
        self.last_print_time_recv = 0

        self.frame_event = asyncio.Event()
        self.frame_show_event = asyncio.Event()
        self.result_event = asyncio.Event()
        self.result_show_event = asyncio.Event()

        self._tasks = []

        self.intrinsics = np.eye(4)
        self.intrinsics3x3_inv = np.eye(3)
        self.extrinsics = np.eye(4)
        self.width = pv_width
        self.height = pv_height
        self._center_ray = np.array([self.width / 2, self.height / 2, 1])

        self.last_print_time_recv = time.time()
        self.count_recv = 0
        self.avai_pose_count = 0
        self.last_frame_time = 0

    def stop(self) -> None:
        self.enable = False

    async def run(self, host: str, handlers: List[Callable[["VideoProcessApp"], Any]] = []) -> None:
        self.host = host

        self.enable = True

        # if tk_root is None:
        #     print("UI not detected, use keyboard to stop the program")

        #     def on_press(key):
        #         self.enable = key != keyboard.Key.esc
        #         return self.enable

        #     self.keyboard_listener = keyboard.Listener(on_press=on_press)
        #     self.keyboard_listener.start()

        # else:
        #     print("UI detected, use the stop button to stop the program")
        self.keyboard_listener = None

        global g_on_close
        # set the on close method to stop the program
        g_on_close = self.stop


        hl2ss_lnm.start_subsystem_pv(host, VIDEO_PORT)

        cal_data = self.get_calibration_data()
        self.intrinsics, self.extrinsics = hl2ss_3dcv.pv_fix_calibration(cal_data.intrinsics, np.eye(4, 4, dtype=np.float32))
        self.intrinsics3x3_inv: np.ndarray = np.linalg.inv(self.intrinsics[:3, :3])
        # print(self.intrinsics, self.extrinsics)

        # self.intrinsics = np.array([[493.36,0,0,0],
        #                     [0,493.15,0,0],
        #                     [314.6,171.67,1,0],
        #                     [0,0,0,1]])
        # self.extrinsics = np.array([[1,0,0,0],
        #                     [0,-1,0,0],
        #                     [0,0,-1,0],
        #                     [0,0,0,1]])
        # self.intrinsics3x3_inv = np.linalg.inv(self.intrinsics[:3, :3])

        self.producer = hl2ss_mp.producer()
        self.producer.configure(VIDEO_PORT, hl2ss_lnm.rx_pv(
            host,
            control_port=VIDEO_PORT,
            stream_port=VIDEO_UDP_PORT,
            width=pv_width,
            height=pv_height,
            framerate=pv_framerate,
            profile=hl2ss.VideoProfile.H264_MAIN,
            recv_callback=recv_data_callback))

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()

        self.producer.initialize(VIDEO_PORT, buffer_elements)
        self.producer.start(VIDEO_PORT)
        self.video_sink = consumer.create_sink(self.producer, VIDEO_PORT, manager, None)
        self.video_sink.get_attach_response()
        while (self.video_sink.get_buffered_frame(0)[0] != 0):
            pass

        # self.client = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE) # Create hl2ss client object
        self.client = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE, hl2ss.IPCUDPPort.UNITY_MESSAGE_QUEUE)
        self.client.open() # Connect to HL2

        # first, send meta info: width, height, framerate, self ip, projection matrix
        metadata = {
            "width": pv_width,
            "height": pv_height,
            "fps": pv_framerate,
            "ip": self_ip(),
            "projection": (self._center_ray @ self.intrinsics3x3_inv).tolist(),
        }

        self.send_data(json.dumps(metadata), command_type=0xFFFFFFFD, safe=True)

        global included_classes, scenario, exclude_classes
        # load_model(scenario)
        # CLASSES = get_classes()
        # included_classes = [c for c in CLASSES if c not in exclude_classes]

        self.model_process = ToyProcess(load_model, get_recognition, scenario)
        self.plot_process = ToyProcess(None, draw_recognition)

        self._tasks = [asyncio.create_task(handler(self)) for handler in handlers]
        self._tasks.append(asyncio.create_task(self.receive_frame()))
        self._tasks.append(asyncio.create_task(self.send_heartbeat()))

        await asyncio.gather(*self._tasks, return_exceptions=True)


    def get_frame_callback(self, data: _packet) -> None:
        self.latest_frame: np.ndarray = data.payload.image
        self.latest_timestamp = data.timestamp

        if data.pose[3][3] != 0:
            self.latest_pose = data.pose
            self.avai_pose_count += 1
        else:
            self.latest_pose = None

        self.frame_event.set()
        self.frame_show_event.set()

        self.count_recv += 1
        if self.count_recv > 0 and time.time() - self.last_print_time_recv > print_threshold:
            print(f"Receive frame: {self.count_recv / (time.time() - self.last_print_time_recv):.2f} fps")
            # framerate = self.count_recv / (time.time() - self.last_print_time_recv)
            # self.send_backendStat(framerate, bandwidth)
            self.count_recv = 0
            self.avai_pose_count = 0
            self.last_print_time_recv = time.time()


    async def receive_frame(self) -> None:
        last_stamp = -1
        last_timestamp = -1

        while self.enable:
            stamp, data = self.video_sink.get_most_recent_frame()
            if data is not None and data.payload.image is not None and stamp > last_stamp and data.timestamp > last_timestamp:
                last_stamp = stamp
                last_timestamp = data.timestamp
                self.get_frame_callback(data)

            await asyncio.sleep(0)


    async def send_heartbeat(self) -> None:
        while self.enable:
            self.send_data(hl2ss.HEARTBEAT_PACKAGE, safe=True)
            await asyncio.sleep(1)


    async def close(self) -> None:
        print("Closing")
        for task in self._tasks:
            task.cancel()
        # wait for all tasks to be cancelled
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self.model_process.stop()
        self.plot_process.stop()

        if self.keyboard_listener is not None:
            self.keyboard_listener.join()

        self.video_sink.detach()
        self.producer.stop(VIDEO_PORT)
        print(f'Stopped {VIDEO_PORT}')

        # hl2ss_lnm.stop_subsystem_pv(self.host, VIDEO_PORT)
        self.client.close()

        cv2.destroyAllWindows()

        if tk_root is not None:
            # close the UI in the main thread
            tk_root.after(0, tk_root.quit)
            print("UI closed")


    def send_data(self, data: Union[bytes, str, umq_command_buffer], command_type: int = 0xFFFFFFFE, safe: bool = False) -> None:
        if isinstance(data, umq_command_buffer):
            buffer = data
            # ignore command_type
        else:
            if isinstance(data, str):
                data = data.encode('utf-8')
            buffer = command_buffer() # Create command buffer
            buffer.debug_message(data, command_type) # Append send_debug_message command

        if safe:
            self.client.push(buffer) # Send commands in buffer to the Unity app
        else:
            self.client.push_udp(buffer)


    def get_calibration_data(self) -> hl2ss._Mode2_PV:
        data = hl2ss_lnm.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, pv_width, pv_height, pv_framerate)
        return data


    def get_pos_rot(self, pose: List[List[float]]) -> Tuple[List[float], List[float]]:
        pose: np.ndarray = np.array(pose)

        # position is the last row of the pose matrix
        pos = pose[3, :3].tolist()
        pos[2] = -pos[2]  # z is inverted

        # rotation matrix
        pixel_direction = self._center_ray @ self.intrinsics3x3_inv @ pose[:3, :3]
        pixel_direction /= np.linalg.norm(pixel_direction)
        rot = pixel_direction.tolist()
        rot[0] = -rot[0]  # x is inverted
        rot[1] = -rot[1]  # y is inverted

        return pos, rot


# Display video ---------------------------------------------------------------
async def render_video(app: VideoProcessApp) -> None:
    while app.enable:
        try:
            await asyncio.wait_for(app.frame_show_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.frame_show_event.is_set():
            continue
        
        app.frame_show_event.clear()

        frame = app.latest_frame

        cv2.imshow("Input", frame)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()


async def save_dummy_pic(app: VideoProcessApp) -> None:
    num_dummy = 0
    while app.enable:
        try:
            await asyncio.wait_for(app.frame_show_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.frame_show_event.is_set():
            continue
        
        app.frame_show_event.clear()

        frame = app.latest_frame

        cv2.imshow("Input", frame)

        # if pressed 's', save the frame
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f"dummy-pic/dummy{num_dummy}.png", frame)
            num_dummy += 1


# Recognition -----------------------------------------------------------------
total_time_rec = 0
count_rec = 0
last_print_time_rec = 0

async def run_recognition(app: VideoProcessApp) -> None:
    global exclude_classes, included_classes, g_pause

    while app.enable:
        try:
            await asyncio.wait_for(app.frame_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.frame_event.is_set():
            continue

        app.frame_event.clear()
        try:
            frame = app.latest_frame
            pose = app.latest_pose
            timestamp = app.latest_timestamp

            if g_pause:
                result = {
                    "masks": [],
                    "mask_contours": [],
                    "boxes": [],
                    "geometry_center": [],
                    "scores": [],
                    "class_names": [],
                    "labels": [],
                }
            else: 
                start_time = time.time()

                # result = get_recognition(frame, score_threshold = 0.45, top_k = 40,
                #                         filter_objects=included_classes,)
                result = await app.model_process.get_result(image=frame, score_threshold = 0.45, top_k = 40,
                                        exclude_objects=exclude_classes)
                end_time = time.time()

            global total_time_rec, count_rec, last_print_time_rec
            total_time_rec += end_time - start_time
            count_rec += 1

            if count_rec > 0 and time.time() - last_print_time_rec > print_threshold:
                # print both average latency and overall framerate
                print(f"Recognition: latency {total_time_rec / count_rec:.2f} s ({count_rec / total_time_rec:.2f} fps), framerate {count_rec / (time.time() - last_print_time_rec):.2f} fps")
                count_rec = 0
                total_time_rec = 0
                last_print_time_rec = time.time()

            # set the (frame, result, pose, timestamp) tuple
            app.latest_result = (frame, result, pose, timestamp)
            app.result_show_event.set()
            app.result_event.set()

        except Exception as e:
            print(traceback.format_exc())


# Send recognition ------------------------------------------------------------
total_time_send = 0
count_send = 0
last_print_time_send = 0

async def send_recognition(app: VideoProcessApp, lookback: bool = False) -> None:
    while app.enable:
        try:
            await asyncio.wait_for(app.result_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.result_event.is_set():
            continue

        app.result_event.clear()

        try:
            start_time = time.time()
            # get the frame and result
            (frame, result, pose, timestamp) = app.latest_result

            # no labels as they are intermediate data
            # no masks as they are too large to send
            needed_fields = [
                "mask_contours",
                "boxes",
                "geometry_center",
                "scores",
                "class_names",
            ]
            # needed_fields = []
            result_tosend = {field: result[field] for field in needed_fields}

            # add camera position
            # if pose is not None:
            #     pos, rot = app.get_pos_rot(pose)
            #     result_tosend["position"] = pos
            #     result_tosend["rotation"] = rot

            # serialize the result
            # sresult: bytes
            global g_values
            with color_lock:
                color_to_send = (*g_color_mapping[g_values['primary_color']], g_values['primary_alpha'])

            with design_lock:
                design_to_send = g_values['primary_design']

            with contour_lock:
                thickness_to_send = g_values['primary_contour_thickness']

            with label_lock:
                global g_label_size, g_text_size
                result_tosend["label_size"] = g_label_size
                result_tosend["text_size"] = g_text_size
                label_color_to_send = (*g_color_mapping[g_values['primary_label_color']], g_values['primary_label_alpha'])

            name_mapping: Dict[str, str] = {}
            with metadata_lock:
                global g_metadata, g_use_diff_vis, g_diff_colors, g_diff_designs
                global g_name_mapping, g_diff_thicknesses, g_diff_label_colors
                name_mapping = g_name_mapping
                if g_use_diff_vis:
                    color_to_send = g_diff_colors
                    design_to_send = g_diff_designs
                    thickness_to_send = g_diff_thicknesses
                    label_color_to_send = g_diff_label_colors

            sresult = serialize(
                result_tosend,
                colors = color_to_send,
                augmentations = design_to_send,
                thicknesses = thickness_to_send,
                label_colors = label_color_to_send,
                timestamp = timestamp,
                name_mapping = name_mapping,
            )

            # send the message
            app.send_data(sresult)

            end_time = time.time()
            global total_time_send, count_send, last_print_time_send
            total_time_send += end_time - start_time
            count_send += 1

            if count_send > 0 and time.time() - last_print_time_send > print_threshold:
                # print both average latency and overall framerate
                print(f"Send Recognition: latency {total_time_send / count_send:.2f} s, framerate {count_send / (time.time() - last_print_time_send):.2f} fps")
                count_send = 0
                total_time_send = 0
                last_print_time_send = time.time()
        
        except Exception as e:
            print(traceback.format_exc())


# Draw recognition ------------------------------------------------------------
total_time_show = 0
count_show = 0
last_print_time_show = 0

async def show_recognition(app: VideoProcessApp) -> None:
    while app.enable:
        try:
            await asyncio.wait_for(app.result_show_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.result_show_event.is_set():
            continue

        app.result_show_event.clear()

        start_time = time.time()
        # get the frame and result
        (frame, result, pose, timestamp) = app.latest_result

        image = await app.plot_process.get_result(image = frame, result = result, alpha = 0.45,
                                    draw_contour = True, black = False,
                                    draw_mask = False, draw_box = False,
                                    draw_text = True, draw_score = False,
                                    draw_center = True, lv_color=(0, 255, 255), contour_thickness=8)
        
        cv2.imshow("Recognition", image)
        cv2.waitKey(1)

        end_time = time.time()
        global total_time_show, count_show, last_print_time_show
        total_time_show += end_time - start_time
        count_show += 1

        if count_show > 0 and time.time() - last_print_time_show > print_threshold:
            # print both average latency and overall framerate
            print(f"Show Recognition: latency {total_time_show / count_show:.2f} s ({count_show / total_time_show:.2f} fps), framerate {count_show / (time.time() - last_print_time_show):.2f} fps")
            count_show = 0
            total_time_show = 0
            last_print_time_show = time.time()


async def test_pose(app: VideoProcessApp) -> None:
    while app.enable:
        try:
            await asyncio.wait_for(app.frame_event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # continue to check if app.enable is False
            pass

        if not app.enable:
            break

        if not app.frame_event.is_set():
            continue

        app.frame_event.clear()
        try:
            pose = app.latest_pose

            if pose is not None:
                pos, rot = app.get_pos_rot(pose)

                result = {'position': pos, 'rotation': rot}
                sresult = serialize(result)
                app.send_data(sresult, safe=True)

        except Exception as e:
            print(traceback.format_exc())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backend entry point")
    parser.add_argument("--host", "-ip", help="ip address of signaler/sender instance")
    args = parser.parse_args()
    host = args.host or "localhost"

    app = VideoProcessApp()

    # run UI
    ui_thread = threading.Thread(target=run_ui)
    ui_thread.daemon = True
    ui_thread.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.run(host, handlers=[
        # render_video,
        run_recognition,
        send_recognition,
        show_recognition,
        # test_pose,
        # save_dummy_pic,
    ]))
    loop.run_until_complete(app.close())
    loop.close()

    if ui_thread.is_alive():
        ui_thread.join()