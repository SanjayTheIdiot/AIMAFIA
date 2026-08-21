import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Low-latency settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Try to keep the camera buffer tiny
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Preview only
    display = cv2.flip(frame, 1)

    cv2.imshow("AI Mafia Camera", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()