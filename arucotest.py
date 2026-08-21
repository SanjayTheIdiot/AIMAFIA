import cv2
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

# ArUco ID -> Player name
PLAYER_NAMES = {
    0: "P1",
    1: "P2",
    2: "P3",
    3: "P4",
    4: "P5",
    5: "P6"
}

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Could not open Logitech camera.")
    exit()

print("Camera started.")

# ============================================================
# ARUCO
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

# ============================================================
# SEAT DATA
# ============================================================

seats = {}

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Reset current detections
    current_seats = {}

    if ids is not None:

        for marker_corners, marker_id in zip(corners, ids):

            marker_id = int(marker_id)

            # Only accept IDs 0-5
            if marker_id not in PLAYER_NAMES:
                continue

            points = marker_corners[0]

            # Center of marker
            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))

            player = PLAYER_NAMES[marker_id]

            # Save seat information
            current_seats[player] = {
                "id": marker_id,
                "x": center_x,
                "y": center_y
            }

            # Draw marker
            cv2.aruco.drawDetectedMarkers(
                frame,
                [marker_corners],
                np.array([[marker_id]], dtype=np.int32)
            )

            # Draw center
            cv2.circle(
                frame,
                (center_x, center_y),
                6,
                (0, 255, 0),
                -1
            )

            # Draw player name
            cv2.putText(
                frame,
                player,
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Draw coordinates
            cv2.putText(
                frame,
                f"({center_x},{center_y})",
                (center_x + 10, center_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

    # Store latest detections
    seats = current_seats

    # ========================================================
    # DRAW CONNECTIONS BETWEEN SEATS
    # ========================================================

    detected_points = []

    for player in ["P1", "P2", "P3", "P4", "P5", "P6"]:

        if player in seats:

            x = seats[player]["x"]
            y = seats[player]["y"]

            detected_points.append((player, x, y))

    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    status_y = 25

    for player in ["P1", "P2", "P3", "P4", "P5", "P6"]:

        if player in seats:

            text = f"{player}: DETECTED"
            text_color = (0, 255, 0)

        else:

            text = f"{player}: NOT DETECTED"
            text_color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2
        )

        status_y += 22

    # Mirror ONLY the preview
    display = cv2.flip(frame, 1)

    cv2.imshow(
        "AI Mafia - Six Seat Tracker",
        display
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")