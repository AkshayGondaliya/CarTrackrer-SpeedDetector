class Config():
    def __init__(self):
        self.boundry_box_width = 2
        self.font_scale = 2
        self.font_thikness = 2
        self.imgsz = 640
        self.classes = 2 # to detect car
        self.conf = 0.40 # confidence threshold
        self.iou_thres = 0.45 # IOU threshold
        self.hide_label = True # Hide labels on bounding boxes
        self.hide_conf = True # Hide confidence score on bounding boxes
