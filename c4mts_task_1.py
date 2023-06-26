from ultralytics import YOLO

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key = "hwPcQ0jxgrwNqfr9MuJ85ORtY",
  project_name = "general",
  workspace="naagar"
)

# MODEL
# model=YOLO("yolov8n.pt")
# model=YOLO("yolov8s.pt")
# model=YOLO("yolov8m.pt")
model=YOLO("yolov8l.pt")

# runs/detect/train2
# touch data.yaml  to create an empty yaml file

model.train(batch=48, data="data_det.yaml", epochs=180, dropout=0.30, weight_decay=0.0002, imgsz=720, momentum=0.982, patience=200, #)
    degrees= 90 , scale = 0.10 , shear=10, perspective=0.0002, flipud=0.25, fliplr=0.25, mosaic=0.8, mixup=0.10, 
    )

# testing 
# source = './datasets/c4mts_1/test/'
# model = YOLO('runs/detect/train2/weights/best.pt')
# results = model.predict(source, conf=0.0001, save_txt=True, save_conf=True,)
source = './datasets/task1test/'
results = model.predict(source, save=True, conf=0.0001, save_txt=True, save_conf=True,)


# Predction mode
# inputs = [img, img]  # list of numpy arrays
# results = model(inputs)  # list of Results objects

# for result in results:
    # boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # probs = result.probs  # Class probabilities for classification outputs
# Key Value   Description
# hsv_h   0.015   image HSV-Hue augmentation (fraction)
# hsv_s   0.7 image HSV-Saturation augmentation (fraction)
# hsv_v   0.4 image HSV-Value augmentation (fraction)
# degrees 0.0 image rotation (+/- deg)
# transl  0.1 image translation (+/- fraction)
# scale   0.5 image scale (+/- gain)
# shear   0.0 image shear (+/- deg)
# perspe  0.0 image perspective (+/- fraction), range 0-0.001
# flipud  0.0 image flip up-down (probability)
# fliplr  0.5 image flip left-right (probability)
# mosaic  1.0 image mosaic (probability)
# mixup   0.0 image mixup (probability)
# copypaste  0.0 segment copy-paste (probability)