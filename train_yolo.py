import torch
from ultralytics import YOLO

# Check if MPS is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Check if the model and data are properly loaded
print(f"Model loaded: {model}")

# Increase number of workers for data loading
data_config = 'taco.yaml'
epochs = 100
imgsz = 640
batch_size = 16  # Adjust based on your system memory

# Set the number of data loader workers
num_workers = 8  # Adjust based on the number of CPU cores

# Train the model
model.train(data=data_config, epochs=epochs, imgsz=imgsz, batch=batch_size, workers=num_workers, device=device)

# Save the trained model
model.save('best_model.pt')