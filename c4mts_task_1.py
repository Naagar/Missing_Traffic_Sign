from ultralytics import YOLO

# MODEL
# model=YOLO("yolov8n.pt")
# model=YOLO("yolov8s.pt")
# model=YOLO("yolov8m.pt")
model=YOLO("yolov8l.pt")


# touch data.yaml  to create an empty yaml file 

model.train(batch=12, data="data.yaml", epochs=200, dropout=0.30, weight_decay=0.0002, imgsz=720, momentum=0.982, patience=200,)

# testing 
source = './datasets/task1test/'
results = model.predict(source, save=True, conf=0.10, save_txt=True, save_conf=True,)


# Predction mode
# inputs = [img, img]  # list of numpy arrays
# results = model(inputs)  # list of Results objects

# for result in results:
    # boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # probs = result.probs  # Class probabilities for classification outputs