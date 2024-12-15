import cv2
import numpy as np
from PIL import Image

# Load YOLO model (using OpenCV DNN module)
net = cv2.dnn.readNetFromDarknet("cross-hands-tiny.cfg", "cross-hands-tiny.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]




# Define desk regions
bbox_annotations = [
    {"label": "TANVIR", "coordinates": {"x": 604.922480620155, "y": 557.0, "width": 126.0, "height": 180.0}},
    {"label": "SHAFAYET", "coordinates": {"x": 751.922480620155, "y": 531.5, "width": 148.0, "height": 83.0}},
    {"label": "TOUFIQ", "coordinates": {"x": 875.922480620155, "y": 517.0, "width": 92.0, "height": 114.0}},
    {"label": "MUFRAD", "coordinates": {"x": 1035.422480620155, "y": 534.0, "width": 141.0, "height": 114.0}},
    {"label": "IMRAN", "coordinates": {"x": 1207.922480620155, "y": 550.5, "width": 126.0, "height": 111.0}},
    {"label": "EMON", "coordinates": {"x": 1340.922480620155, "y": 565.0, "width": 114.0, "height": 140.0}},
    {"label": "ANIK", "coordinates": {"x": 1173.422480620155, "y": 671.5, "width": 129.0, "height": 175.0}},
    {"label": "FAISAL", "coordinates": {"x": 891.922480620155, "y": 651.0, "width": 102.0, "height": 184.0}}
]

# Convert bounding box annotations to desk regions
desk_regions = []
for annotation in bbox_annotations:
    coords = annotation["coordinates"]
    desk_regions.append({
        "name": annotation["label"],
        "region": (
            int(coords["x"] - coords["width"] / 2),
            int(coords["y"] - coords["height"] / 2),
            int(coords["x"] + coords["width"] / 2),
            int(coords["y"] + coords["height"] / 2)
        )
    })

# Function to detect which desk the hand is in based on coordinates
def map_to_desk(x, y):
    for desk in desk_regions:
        x_min, y_min, x_max, y_max = desk["region"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return desk["name"]
    return "Unknown Desk"

# Process video
video_path = "desk_video.mp4"
cap = cv2.VideoCapture(video_path)

output_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id == 0 and confidence > 0.5:  # Class 0 is 'hand' in this custom model
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Map hand to desk
                desk_name = map_to_desk(center_x, center_y)

                # Annotate frame
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(frame, desk_name, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save frame if hand is detected in a desk region
                if desk_name != "Unknown Desk":
                    output_frames.append(frame.copy())

    # Show the frame for debugging
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save output
if output_frames:
    stitched_image = np.concatenate(output_frames, axis=1)  # Horizontally concatenate frames
    stitched_image_pil = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    stitched_image_pil.save("stitched_output_yolo.png")
    print("Processing complete. Stitched output saved as 'stitched_output_yolo.png'.")
else:
    print("No frames with hands raised detected.")
