import cv2
import mediapipe as mp
import numpy as np
import json

# Desk regions
bbox_annotations = [
    {"label": "TANVIR", "coordinates": {"x": 604.92, "y": 557.0, "width": 126.0, "height": 180.0}},
    {"label": "SHAFAYET", "coordinates": {"x": 751.92, "y": 531.5, "width": 148.0, "height": 83.0}},
    {"label": "TOUFIQ", "coordinates": {"x": 875.92, "y": 517.0, "width": 92.0, "height": 114.0}},
    {"label": "MUFRAD", "coordinates": {"x": 1035.42, "y": 534.0, "width": 141.0, "height": 114.0}},
    {"label": "IMRAN", "coordinates": {"x": 1207.92, "y": 550.5, "width": 126.0, "height": 111.0}},
    {"label": "EMON", "coordinates": {"x": 1340.92, "y": 565.0, "width": 114.0, "height": 140.0}},
    {"label": "ANIK", "coordinates": {"x": 1173.42, "y": 671.5, "width": 129.0, "height": 175.0}},
    {"label": "FAISAL", "coordinates": {"x": 891.92, "y": 651.0, "width": 102.0, "height": 184.0}},
]

# Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open the video file
video_path = "desk_video.mp4"  # Update if necessary
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file not found or could not be opened.")
    exit()

# Rescale bounding boxes based on the frame size
def rescale_bboxes(frame_width, frame_height):
    for desk in bbox_annotations:
        coords = desk["coordinates"]
        coords["x"] = int(coords["x"] * frame_width / 1920)  # Assuming original video is 1920x1080
        coords["y"] = int(coords["y"] * frame_height / 1080)
        coords["width"] = int(coords["width"] * frame_width / 1920)
        coords["height"] = int(coords["height"] * frame_height / 1080)

# Determine which desk the hand is in
def get_desk_for_hand(x, y):
    for desk in bbox_annotations:
        desk_x, desk_y = desk["coordinates"]["x"], desk["coordinates"]["y"]
        desk_width, desk_height = desk["coordinates"]["width"], desk["coordinates"]["height"]
        if desk_x <= x <= desk_x + desk_width and desk_y <= y <= desk_y + desk_height:
            return desk["label"]
    return None

# Output variables
hand_raises = {}
frame_count = 0
processed_frames = []

# Read the first frame to rescale bounding boxes
ret, first_frame = cap.read()
if ret:
    h, w, _ = first_frame.shape
    rescale_bboxes(w, h)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            desk_label = get_desk_for_hand(wrist_x, wrist_y)
            if desk_label:
                hand_raises[frame_count] = desk_label

                # Annotate the frame
                cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 255, 0), -1)
                cv2.putText(frame, desk_label, (wrist_x, wrist_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_frames.append(frame)

cap.release()
hands.close()

# Save hand raises as JSON
with open("hand_raises.json", "w") as f:
    json.dump(hand_raises, f)

# Save a summary image (stitched frames)
stitched_image = cv2.vconcat(processed_frames[:10]) if len(processed_frames) > 0 else None
if stitched_image is not None:
    cv2.imwrite("summary_image.jpg", stitched_image)

print("Processing complete. Detections saved in hand_raises.json.")
