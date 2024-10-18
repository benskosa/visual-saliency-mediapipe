#------------------------------------------------------------------------------
# This script receives encoded video from the HoloLens cameras and plays it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import threading
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import time
import asyncio
import argparse
from utils.serializer import serialize
from utils.timeutils import current_time
from yolo.api import async_get_recognition, async_draw_recognition, get_recognition
from typing import Tuple, Dict, Any, Union, List, Callable

try:
    import ujson as json
except ImportError:
    print("ujson not found, using json")
    import json

# Settings --------------------------------------------------------------------

# Ports
ports = [
    # hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    # hl2ss.StreamPort.RM_VLC_LEFTLEFT,
    # hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    # hl2ss.StreamPort.RM_VLC_RIGHTRIGHT,
    #hl2ss.StreamPort.RM_DEPTH_AHAT,
    # hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    hl2ss.StreamPort.PERSONAL_VIDEO,
    # hl2ss.StreamPort.RM_IMU_ACCELEROMETER,
    # hl2ss.StreamPort.RM_IMU_GYROSCOPE,
    # hl2ss.StreamPort.RM_IMU_MAGNETOMETER,
    # hl2ss.StreamPort.MICROPHONE,
    # hl2ss.StreamPort.SPATIAL_INPUT,
    # hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
]


# PV parameters
pv_width     = 640
pv_height    = 360
pv_framerate = 30

# Maximum number of frames in buffer
buffer_elements = 150

#------------------------------------------------------------------------------

class VideoProcessApp:
    def __init__(self):
        self.latest_frame: np.ndarray = None
        # a result is a tuple of (frame, result)
        self.latest_result: Tuple[np.ndarray, Dict[str, Any]] = None
        self.latest_output_video: np.ndarray = None
        self.last_frame_time_clear = 0

        self.count_recv = 0
        self.last_print_time_recv = 0

        self.frame_event = asyncio.Event()
        self.result_event = asyncio.Event()
        self.result_show_event = asyncio.Event()
    
        self._tasks = []

    async def run(self, host: str, handlers: List[Callable[["VideoProcessApp"], Any]] = []) -> None:
        enable = True

        def on_press(key):
            nonlocal enable
            enable = key != keyboard.Key.esc
            return enable

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        self._tasks = [asyncio.create_task(handler(self)) for handler in handlers]





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backend entry point")
    parser.add_argument("--host", "-ip", help="ip address of signaler/sender instance")
    args = parser.parse_args()
    host = args.host or "localhost"

    
    
    
    
    
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem if PV is selected ------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTLEFT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTLEFT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTRIGHT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTRIGHT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_IMU_ACCELEROMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER))
    producer.configure(hl2ss.StreamPort.RM_IMU_GYROSCOPE, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_GYROSCOPE))
    producer.configure(hl2ss.StreamPort.RM_IMU_MAGNETOMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER))
    producer.configure(hl2ss.StreamPort.MICROPHONE, hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER))

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sinks = {}

    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()
        while (sinks[port].get_buffered_frame(0)[0] != 0):
            pass
        print(f'Started {port}')
        
    # Create Display Map ------------------------------------------------------
    def display_pv(port, payload):
        if (payload.image is not None and payload.image.size > 0):
            cv2.imshow(hl2ss.get_port_name(port), payload.image)

    def display_basic(port, payload):
        if (payload is not None and payload.size > 0):
            cv2.imshow(hl2ss.get_port_name(port), payload)

    def display_depth_lt(port, payload):
        cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 8) # Scaled for visibility
        cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_depth_ahat(port, payload):
        if (payload.depth is not None and payload.depth.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 64) # Scaled for visibility
        if (payload.ab is not None and payload.ab.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_null(port, payload):
        pass

    DISPLAY_MAP = {
        hl2ss.StreamPort.RM_VLC_LEFTFRONT     : display_basic,
        hl2ss.StreamPort.RM_VLC_LEFTLEFT      : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT    : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTRIGHT    : display_basic,
        hl2ss.StreamPort.RM_DEPTH_AHAT        : display_depth_ahat,
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW   : display_depth_lt,
        hl2ss.StreamPort.PERSONAL_VIDEO       : display_pv,
        hl2ss.StreamPort.RM_IMU_ACCELEROMETER : display_null,
        hl2ss.StreamPort.RM_IMU_GYROSCOPE     : display_null,
        hl2ss.StreamPort.RM_IMU_MAGNETOMETER  : display_null,
        hl2ss.StreamPort.MICROPHONE           : display_null,
        hl2ss.StreamPort.SPATIAL_INPUT        : display_null,
        hl2ss.StreamPort.EXTENDED_EYE_TRACKER : display_null,
    }

    # Send text ---------------------------------------------------------------
    class command_buffer(hl2ss.umq_command_buffer):
        # Command structure
        # id:     u32 (4 bytes)
        # size:   u32 (4 bytes)
        # params: size bytes

        # Send string to Visual Studio debugger
        def debug_message(self, text):
            # Command id: 0xFFFFFFFE
            # Command params: string encoded as utf-8
            if isinstance(text, str):
                self.add(0xFFFFFFFE, text.encode('utf-8'))
            else:
                self.add(0xFFFFFFFE, text)

        # See hl2ss_rus.py and the unity_sample scripts for more examples.

    client = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE) # Create hl2ss client object
    client.open() # Connect to HL2

    # first, send the image size
    buffer = command_buffer() # Create command buffer
    buffer.add(0xFFFFFFFD, json.dumps({"width": pv_width, "height": pv_height, "fps": pv_framerate}).encode('utf-8'))
    client.push(buffer) # Send commands in buffer to the Unity app
    response = client.pull(buffer) # Receive response from the Unity app (4 byte integer per command)

    # Main loop ---------------------------------------------------------------
    last_stamps = {port: -1 for port in ports}
    while enable:
        imgL = imgR = None
        for port in ports:
            stamp, data = sinks[port].get_most_recent_frame()
            if data is not None and stamp > last_stamps[port]:
                last_stamps[port] = stamp
                image = data.payload.image
                cv2.imshow(hl2ss.get_port_name(port), image)
                cv2.waitKey(1)
                # result = get_recognition(image, score_threshold=0.5, top_k=20)

            # buffer = command_buffer() # Create command buffer
            # buffer.debug_message(str(ans)) # Append send_debug_message command
            # client.push(buffer) # Send commands in buffer to the Unity app
            # response = client.pull(buffer) # Receive response from the Unity app (4 byte integer per command)

    # Stop streams ------------------------------------------------------------
    for port in ports:
        sinks[port].detach()
        producer.stop(port)
        print(f'Stopped {port}')

    client.close()

    # Stop PV Subsystem if PV is selected -------------------------------------
    if hl2ss.StreamPort.PERSONAL_VIDEO == port:
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
