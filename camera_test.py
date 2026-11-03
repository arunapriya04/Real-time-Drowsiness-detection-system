import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

if not cap.isOpened():
    print("Cannot open camera")
else:
    print("Camera opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Frame captured")
        print(f"Frame shape: {frame.shape}")
    else:
        print("Cannot capture frame")
    cap.release()