import cv2
from tracker import *
import torch
from time import time
from config import Config
# Create tracker object

tracker = EuclideanDistTracker()
config=Config()

cap = cv2.VideoCapture("highway.mp4")
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model.classes = config.classes # to detect car l
model.conf = config.conf # confidence threshold
model.iou_thres = config.iou_thres # IOU threshold
model.hide_label = config.hide_label # Hide labels on bounding boxes
model.hide_conf = config.hide_conf # Hide confidence score on bounding boxes

rectengle_width = config.boundry_box_width # square boundry line width
font_scale = config.font_scale # font size over the detected car
font_thikness = config.font_thikness # font thikness
ok=0
while True:
    ret, frame = cap.read()
    if frame is not None:
	    ok+=1
	    height, width, _ = frame.shape
	    fps=cap.get(cv2.CAP_PROP_FPS)

	    # Extract Region of interest
	    roi = frame[300: 720,300: 1100]
	    print("roi shape===>",roi.shape)
	    results = model(roi)
	    car_count = len(results.pandas().xyxy[0])
	    if car_count > 0:
	    	det = results.pandas().xyxy[0].to_dict(orient="records")
	    	detections=[]
	    	for result in det:
	    		x = int(result['xmin'])
	    		y = int(result['ymin'])
	    		x2 = int(result['xmax'])
	    		y2 = int(result['ymax'])
	    		w=x2-x

	    		h=y2-y
	    		detections.append([x, y, w, h])
	    	boxes_ids=tracker.update(detections,fps)
	    	for box_id in boxes_ids:
	    		x, y, w, h, id,frame_of_car,km_pr_hour = box_id
	    		cv2.line(frame, (0, 650), (1200, 650), (0, 0, 255), thickness=3)
	    		cv2.line(frame, (0, 300), (1200,300), (0, 0, 255), thickness=3)
	    		cv2.putText(roi, str(id)+"_"+str(frame_of_car)+"_"+km_pr_hour, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), font_thikness)
	    		cv2.rectangle(roi,(x, y), (x + w, y + h), (0, 255, 0), rectengle_width)
	    		# cv2.imshow("frame", roi)
	    cv2.imwrite("image/im_"+str(ok)+".jpg",frame)

    else:
        break
