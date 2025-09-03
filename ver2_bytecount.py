from ultralytics import YOLO
import supervision as sv
import cv2
import csv
import time
from collections import defaultdict

model = YOLO("yolo11n.pt")

# Vehicle classes
vehicle_classes = {"car": 2, "motorbike": 3, "bus": 5, "truck": 7}
vehicle_counts = defaultdict(int)

# Initializing ByteTrack
tracker = sv.ByteTrack()

# Horizontal counting line
line_y = 200
crossed_ids = set()

# opening CSV file
csv_file = open("vehicle_line_counts.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Car", "Motorbike", "Bus", "Truck"])

# Opening video file/ live feed
cap = cv2.VideoCapture("sample3.mp4")  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filtering for vehicles
    mask = [cls in vehicle_classes.values() for cls in detections.class_id]
    detections = detections[mask]

    # Tracking objects
    tracks = tracker.update_with_detections(detections)

    # Drawing counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

    for xyxy, cls_id, track_id in zip(tracks.xyxy, tracks.class_id, tracks.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        label = [k for k, v in vehicle_classes.items() if v == cls_id][0]

        # Drawing box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Counting line crossing
        if line_y - 5 < cy < line_y + 5 and track_id not in crossed_ids:
            vehicle_counts[label] += 1
            crossed_ids.add(track_id)

    # To see counts on screen
    y0 = 30
    for i, (veh, count) in enumerate(vehicle_counts.items()):
        cv2.putText(frame, f"{veh}: {count}", (10, y0 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Saving
    if int(time.time()) % 5 == 0:
        csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                             vehicle_counts["car"],
                             vehicle_counts["motorbike"],
                             vehicle_counts["bus"],
                             vehicle_counts["truck"]])
        csv_file.flush()

    cv2.imshow("Vehicle Counter with ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Final counts:", dict(vehicle_counts))



