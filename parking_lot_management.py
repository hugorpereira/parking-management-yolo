import cv2

from ultralytics import solutions
from util import check_offset

# Video capture
cap = cv2.VideoCapture("activities/parking-management-yolo/data/3858833-hd_1280_720_24fps.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking_management.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parkingmanager = solutions.ParkingManagement(
    model="activities/parking-management-yolo/model/best.pt",
    json_file="activities/parking-management-yolo/data/boxes/parking_spots_boxes_24fps.json",
    show=True,
    classes=[3, 4, 5, 8], # car, van, truck, bus
    line_width=1
)

original_json = parkingmanager.json

# Set how many frames will be selected per second
TARGET_FPS = fps
frame_interval = int(fps / TARGET_FPS)

while cap.isOpened():

    ret, im0 = cap.read()
    if not ret:
        break
    
    # Current Frame
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Process only frames in predefined interval
    if current_frame % frame_interval == 0:

        # Check and shift boxes to adjust video movements
        boxes = check_offset(
            current_frame,
            original_boxes=original_json
        )
        
        # If boxes were changed, update parking management object
        if len(boxes) > 0:
            parkingmanager.json = boxes
        
        # Write the results in the new video
        results = parkingmanager(im0)
        video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
