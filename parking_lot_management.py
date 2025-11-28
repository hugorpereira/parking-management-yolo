import cv2

from ultralytics import solutions

# Video capture
# cap = cv2.VideoCapture("data/5587732-hd_1280_720_30fps.mp4")
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
    classes=[3, 4, 5, 8],
    # tracker="bytetrack.yaml",
    # conf=0.3,
    # iou=0.5,
    # verbose=True,
    line_width=1,
)

def shift_points_horizontal(data, shift=5):
    for item in data:
        new_points = []
        for x, y in item["points"]:
            new_points.append([x + shift, y])
        item["points"] = new_points
    return data

original_json = parkingmanager.json

TARGET_FPS = fps
frame_interval = int(fps / TARGET_FPS)
frame_count = 0

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        
        if frame_count >= 10 and frame_count < 50:
            parkingmanager.json = shift_points_horizontal(original_json, 3/(50-10))
        elif frame_count >= 50 and frame_count < 80:
            parkingmanager.json = shift_points_horizontal(original_json, 7/(80-50))
        elif frame_count >= 80 and frame_count < 200:
            parkingmanager.json = shift_points_horizontal(original_json, 5/(200-80))
        elif frame_count >= 200 and frame_count < 300:
            parkingmanager.json = shift_points_horizontal(original_json, -8/(300-200))
        elif frame_count >= 300 and frame_count < 370:
            parkingmanager.json = shift_points_horizontal(original_json, -9/(370-300))
        elif frame_count >= 370 and frame_count < 400:
            parkingmanager.json = shift_points_horizontal(original_json, -9/(400-370))
        elif frame_count >= 400 and frame_count < 500:
            parkingmanager.json = shift_points_horizontal(original_json, -9/(500-400))
        elif frame_count >= 500 and frame_count < 600:
            parkingmanager.json = shift_points_horizontal(original_json, -9/(600-500))
        
        cv2.putText(im0, f'{frame_count}', (20, 20), 0, 1, (255,255,255), 2, lineType=cv2.LINE_AA)
        results = parkingmanager(im0)
        video_writer.write(results.plot_im)  # write the processed frame.
    
    frame_count += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
