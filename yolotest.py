import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

model = YOLO("best.pt")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Real-Time Detection with RealSense', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
