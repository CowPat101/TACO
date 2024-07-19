import json
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

def convert_annotation(data, img_dir, label_dir):
    for item in data['images']:
        print("Printing out item filename: " + str(item['file_name']))
        print("Printing out item id: " + str(item['id']))
        print("Printing out item width: " + str(item['width']))
        print("Printing out item height: " + str(item['height']))
        file_name = item['file_name']
        image_id = item['id']
        width = item['width']
        height = item['height']
        annotations = [a for a in data['annotations'] if a['image_id'] == image_id]
        label_file_name = file_name.replace('.jpg', '.txt')
        
        # Create directories if they don't exist
        label_file_path = os.path.join(label_dir, label_file_name)
        os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
        
        with open(label_file_path, 'w') as f:
            for a in annotations:
                category_id = a['category_id'] - 1  # YOLO format requires zero-indexed category ids
                bbox = a['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                bbox_width = bbox[2] / width
                bbox_height = bbox[3] / height
                f.write(f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Load the annotation data
with open('data/annotations.json') as json_file:
    data = json.load(json_file)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data['images'], test_size=0.2, random_state=42)

# Create separate dictionaries for training and validation annotations
train_annotations = {'images': train_data, 'annotations': [a for a in data['annotations'] if a['image_id'] in [i['id'] for i in train_data]]}
val_annotations = {'images': val_data, 'annotations': [a for a in data['annotations'] if a['image_id'] in [i['id'] for i in val_data]]}

# Convert and save annotations for the training and validation sets
convert_annotation(train_annotations, 'data', 'TACO/yolo_format/labels/train')
convert_annotation(val_annotations, 'data', 'TACO/yolo_format/labels/val')

# Ensure the image directories exist
os.makedirs('TACO/yolo_format/images/train', exist_ok=True)
os.makedirs('TACO/yolo_format/images/val', exist_ok=True)

# Copy training images to the appropriate directory
for item in train_data:
    train_image_path = os.path.join('TACO/yolo_format/images/train', item['file_name'])
    os.makedirs(os.path.dirname(train_image_path), exist_ok=True)
    copyfile(os.path.join('data', item['file_name']), train_image_path)

# Copy validation images to the appropriate directory
for item in val_data:
    val_image_path = os.path.join('TACO/yolo_format/images/val', item['file_name'])
    os.makedirs(os.path.dirname(val_image_path), exist_ok=True)
    copyfile(os.path.join('data', item['file_name']), val_image_path)