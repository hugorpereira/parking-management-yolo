import cv2
vidcap = cv2.VideoCapture('data/3858833-hd_1280_720_24fps.mp4')
success,image = vidcap.read()
count = 0
if success:
  cv2.imwrite("frame%d.jpg" % count, image)
  success,image = vidcap.read()
  print('Read a new frame: ', success)