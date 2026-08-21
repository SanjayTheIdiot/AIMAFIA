import cv2
import time
from ultralytics import YOLO

# Load small, fast model
model = YOLO("yolo11n.pt")

# Logitech camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Could not open Logitech camera.")
    exit()

print("Starting YOLO...")

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Run YOLO
    results = model(
        frame,
        classes=[0],      # Person only
        verbose=False,
        device=0
    )

    # Draw detections
    annotated = results[0].plot()

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Mirror preview only
    display = cv2.flip(annotated, 1)

    cv2.imshow("YOLO Person Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()