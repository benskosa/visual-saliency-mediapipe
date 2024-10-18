from .frame_data_pb2 import FrameData, ImageData
import cv2
import numpy as np

def parse_bytes(data: bytes) -> FrameData:
    frame = FrameData()
    frame.ParseFromString(data)
    return frame

def parse_image(data: bytes) -> np.ndarray:
    frame = parse_bytes(data)
    idata: ImageData = frame.image_data
    width = idata.width
    height = idata.height
    image_bytes = idata.image

    # image_bytes is in YUV420 format
    # convert to BGR
    yuv_data = np.frombuffer(image_bytes, dtype=np.uint8).reshape((int(height * 1.5), width))
    bgr_data = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
    
    return {
        'image': bgr_data,
        'frame_time': frame.frame_time,
        'width': width,
        'height': height,
    } 