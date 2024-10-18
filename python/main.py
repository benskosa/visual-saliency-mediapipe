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

from ultralytics import YOLO

# Settings --------------------------------------------------------------------

# HoloLens address
host = '10.141.12.205'

# Ports
ports = [
    # hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    # hl2ss.StreamPort.RM_VLC_LEFTLEFT,
    # hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    # hl2ss.StreamPort.RM_VLC_RIGHTRIGHT,
    # hl2ss.StreamPort.RM_DEPTH_AHAT,
    # hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    # hl2ss.StreamPort.PERSONAL_VIDEO,
    # hl2ss.StreamPort.RM_IMU_ACCELEROMETER,
    # hl2ss.StreamPort.RM_IMU_GYROSCOPE,
    # hl2ss.StreamPort.RM_IMU_MAGNETOMETER,
    # hl2ss.StreamPort.MICROPHONE,
    # hl2ss.StreamPort.SPATIAL_INPUT,
    # hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
    ]

# PV parameters
pv_width     = 760
pv_height    = 428
pv_framerate = 30

image_size = (640, 480)

# Maximum number of frames in buffer
buffer_elements = 150

# CV Model
# model = YOLO("yolov8l.pt")

#------------------------------------------------------------------------------

if __name__ == '__main__':
    if ((hl2ss.StreamPort.RM_DEPTH_LONGTHROW in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous RM Depth Long Throw and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

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
            # cv2.waitKey(1)

    def display_basic(port, payload):
        if (payload is not None and payload.size > 0):
            # flip 90 degrees
            payload = np.rot90(payload, 3)
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
        print(payload)
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

    last_stamp = -1

    last_print_time = time.time()
    num_frames = 0

    # Main loop ---------------------------------------------------------------
    while enable:
        imgL = imgR = None
        for port in ports:
            stamp, data = sinks[port].get_most_recent_frame()
            if data is not None and stamp > last_stamp:
                last_stamp = stamp
                DISPLAY_MAP[port](port, data.payload)

                num_frames += 1
                if time.time() - last_print_time > 3:
                    print(f"FPS: {num_frames / (time.time() - last_print_time)}")
                    last_print_time = time.time()
                    num_frames = 0

                # results = model(data.payload.image, half=True, verbose=False, retina_masks=True,
                #                 device='cuda:0')  # Inference
                # ans = {}
                # for r in results:
                #     bboxes = r.boxes
                #     for bbox in bboxes:
                #         b_coord = bbox.xyxy[0]  # (left, top, right, bottom) format
                #         c = bbox.cls
                #         obj_name = model.names[int(c)]

                #         obj_key = ""
                #         count = 0
                #         while True:
                #             obj_key = obj_name + str(count)
                #             if obj_key not in ans:
                #                 break 
                #             count += 1

                #         ans[obj_key] = b_coord

                # buffer = command_buffer() # Create command buffer
                # buffer.debug_message(str(ans)) # Append send_debug_message command
                # client.push(buffer) # Send commands in buffer to the Unity app
                # response = client.pull(buffer) # Receive response from the Unity app (4 byte integer per command)

        cv2.waitKey(1)

    # Stop streams ------------------------------------------------------------
    for port in ports:
        sinks[port].detach()
        producer.stop(port)
        print(f'Stopped {port}')

    client.close()

    # Stop PV Subsystem if PV is selected -------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
