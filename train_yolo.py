from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')  # You can also use 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' for larger models

# Train the model on the TACO dataset
model.train(data='taco.yaml', epochs=100, imgsz=640)