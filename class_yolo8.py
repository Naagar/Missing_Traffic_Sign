from ultralytics import YOLO


classes = ['gap-in-median','left-hand-curve','right-hand-curve','side-road-left']
# Load a model
# model = YOLO('data_cls.yaml')  # build a new model from YAML
model = YOLO('yolov8x-cls.pt', task='classify')  # load a pretrained model (recommended for training)
# model = YOLO('data_cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model

model.train(task='classify', dropout=0.45, data='c4mts_1/', classes=classes, epochs=150, imgsz=512, batch=24,
	workers=20, 
	) # exist_ok=True,


# Load a model
# model = YOLO('yolov8n-cls.pt')  # load an official model
# save_dir = './runs/classify/train3/weights/'
# model = YOLO(save_dir+'best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy

# testing 
source = './datasets/c4mts_1/test/'
results = model.predict(source, conf=0.0001, save_txt=True, save_conf=True,)
# results.top1   # top1 accuracy
# results.top5   # top5 accuracy





# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="58t5WW3Y3g5RkMx6eo74")
# project = rf.workspace("c4mts").project("c4mts")
# dataset = project.version(1).download("train_cls_01")
# # !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="████████████████████")
# project = rf.workspace("c4mts").project("c4mts")
# dataset = project.version(1).download("folder")