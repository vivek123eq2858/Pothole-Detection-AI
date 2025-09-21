from ultralytics import YOLO

# Load a small YOLOv8 model
 

# Train on your dataset

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=5, imgsz=640)  # only 5 epochs
