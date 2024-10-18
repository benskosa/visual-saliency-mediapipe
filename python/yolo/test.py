import cv2
import time
from ultralytics import YOLO

# frame = cv2.imread('large_image.jpg')
# Load YOLO model
model = YOLO('pt/yolov8x-seg.pt')
# Try different indices if 0 doesn't work

total_used_time = 0
count = 0

cap = cv2.VideoCapture(0)
# set resolution: 640 * 360
cap.set(3, 640)
cap.set(4, 360)

if not cap.isOpened():
    print("Error: Camera could not be accessed.")
    exit(1)
while True:
    success, frame = cap.read()
    success = True
    if success:
        start = time.perf_counter()
        results = model(frame, verbose=False, half=True)
        end = time.perf_counter()
        total_time = end - start

        count += 1
        total_used_time += total_time
        if count % 30 == 0:
            print(f"Average time: {total_used_time / count:.2f} ({count / total_used_time:.2f} fps)")
            count = 0
            total_used_time = 0

        fps = 1 / total_time if total_time > 0 else 0  # Avoid division by zero
        # Assuming `results.plot()` returns an image
        # annotated_frame = results[0].plot()
        # Display FPS on the frame
        # cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow('frame', annotated_frame)
        # if cv2.waitKey(1) == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
#     else:
#         print("Warning: Frame capture failed.")
#         break
# cap.release()
# cv2.destroyAllWindows()