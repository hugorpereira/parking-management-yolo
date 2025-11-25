import cv2

from ultralytics import solutions

# Video capture
cap = cv2.VideoCapture("data/5587732-hd_1280_720_30fps.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking_management.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parkingmanager = solutions.ParkingManagement(
    model="model/best.pt",
    json_file="data/boxes/parking_spots_boxes.json",
    show=True,
    classes=[3, 4, 5, 8],
    tracker="bytetrack.yaml",
    conf=0.01,
    iou=0.5,
    verbose=True,
    line_width=1,
    max_det=1000,
)

TARGET_FPS = 15
frame_interval = int(fps / TARGET_FPS)

frame_count = 0

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = parkingmanager(im0)
        video_writer.write(results.plot_im)  # write the processed frame.
    
    frame_count += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
