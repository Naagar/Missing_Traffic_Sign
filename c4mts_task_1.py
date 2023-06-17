from ultralytics import YOLO

# MODEL
model=YOLO("yolov8n.pt")

# touch data.yaml

model.train(data="data.yaml",epochs=10)