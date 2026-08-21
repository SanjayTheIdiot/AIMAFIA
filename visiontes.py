import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Maximum distance (pixels) between a person and a seat
# for them to be considered sitting at that seat.
MAX_SEAT_DISTANCE = 180

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Could not open Logitech camera.")
    exit()

# ============================================================
# ARUCO
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

# ============================================================
# YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded.")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # --------------------------------------------------------
    # ARUCO: FIND SEATS
    # --------------------------------------------------------

    corners, ids, rejected = aruco_detector.detectMarkers(frame)

    seats = {}

    if ids is not None:

        for marker_corners, marker_id in zip(corners, ids):

            marker_id = int(marker_id)

            if marker_id not in PLAYER_NAMES:
                continue

            points = marker_corners[0]

            center_x = int(np.mean(points[:, 0]))
            center_y = int(np.mean(points[:, 1]))

            player = PLAYER_NAMES[marker_id]

            seats[player] = {
                "x": center_x,
                "y": center_y,
                "occupied": False
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
                (255, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                player,
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    # --------------------------------------------------------
    # YOLO: FIND PEOPLE
    # --------------------------------------------------------

    results = model(
        frame,
        classes=[0],       # COCO class 0 = person
        conf=0.45,
        verbose=False,
        device=0
    )

    people = []

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(int, xyxy)

            # Person bounding-box center
            person_x = int((x1 + x2) / 2)
            person_y = int((y1 + y2) / 2)

            people.append({
                "x": person_x,
                "y": person_y,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    # --------------------------------------------------------
    # ASSIGN PEOPLE TO SEATS
    # --------------------------------------------------------

    used_people = set()

    for player, seat in seats.items():

        best_person = None
        best_distance = float("inf")
        best_index = None

        for index, person in enumerate(people):

            if index in used_people:
                continue

            dx = person["x"] - seat["x"]
            dy = person["y"] - seat["y"]

            distance = np.sqrt(
                dx * dx + dy * dy
            )

            if distance < best_distance:

                best_distance = distance
                best_person = person
                best_index = index

        # If a person is close enough to the seat
        if (
            best_person is not None
            and best_distance < MAX_SEAT_DISTANCE
        ):

            seat["occupied"] = True

            used_people.add(best_index)

            # Draw line from seat to person
            cv2.line(
                frame,
                (seat["x"], seat["y"]),
                (best_person["x"], best_person["y"]),
                (0, 255, 0),
                2
            )

            # Draw person bounding box
            cv2.rectangle(
                frame,
                (best_person["x1"], best_person["y1"]),
                (best_person["x2"], best_person["y2"]),
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # DISPLAY SEAT STATUS
    # --------------------------------------------------------

    y = 25

    for player in PLAYER_NAMES.values():

        if player in seats:

            if seats[player]["occupied"]:

                text = f"{player}: OCCUPIED"
                color = (0, 255, 0)

            else:

                text = f"{player}: EMPTY"
                color = (0, 0, 255)

        else:

            text = f"{player}: MARKER NOT FOUND"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        y += 22

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display = cv2.flip(frame, 1)

    cv2.imshow(
        "AI Mafia - Vision System",
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