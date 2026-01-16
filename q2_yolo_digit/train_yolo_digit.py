"""
EE 4065 - Embedded Digital Image Processing
Question 2: YOLO Handwritten Digit Detection - Training Script

Student: KAAN ATALAY
ID: 150720057

This script trains a YOLO model for handwritten digit detection (0-9).
The trained model will be converted for ESP32-CAM deployment.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import shutil

# ==================== CONFIGURATION ====================
DATASET_PATH = "digit_dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")
MODEL_OUTPUT = "digit_yolo_model"
IMG_SIZE = 96  # Small size for ESP32
EPOCHS = 100
BATCH_SIZE = 16
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# ==================== DATASET PREPARATION ====================

def create_dataset_structure():
    """Create YOLO dataset directory structure."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATASET_PATH, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_PATH, split, 'labels'), exist_ok=True)
    print("Dataset structure created!")


def create_yaml_config():
    """Create YOLO dataset configuration file."""
    config = {
        'path': os.path.abspath(DATASET_PATH),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 10,  # Number of classes
        'names': CLASSES
    }
    
    yaml_path = os.path.join(DATASET_PATH, 'digit_dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {yaml_path}")
    return yaml_path


def preprocess_image(image_path, output_path, target_size=IMG_SIZE):
    """
    Preprocess handwritten digit image.
    - Convert to grayscale
    - Resize to target size
    - Enhance contrast
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold for better digit extraction
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to locate the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        # Crop digit region
        digit = gray[y:y+h, x:x+w]
    else:
        digit = gray
    
    # Resize to target size
    resized = cv2.resize(digit, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Convert back to BGR for YOLO
    output = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(output_path, output)
    return (x, y, w, h) if contours else None


def create_yolo_label(bbox, img_width, img_height, class_id, output_path):
    """
    Create YOLO format label file.
    Format: class_id x_center y_center width height (all normalized)
    """
    if bbox is None:
        # If no bbox, assume digit covers most of image
        x_center, y_center = 0.5, 0.5
        width, height = 0.8, 0.8
    else:
        x, y, w, h = bbox
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
    
    with open(output_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_synthetic_data(num_samples_per_class=100):
    """
    Generate synthetic handwritten digit images for training.
    This supplements manually collected data.
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    create_dataset_structure()
    
    # Different fonts for variety (use system fonts)
    fonts = []
    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Times.ttc',
        '/Library/Fonts/Arial.ttf',
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fonts.append(ImageFont.truetype(font_path, 60))
            except:
                pass
    
    if not fonts:
        fonts = [ImageFont.load_default()]
    
    print(f"Generating synthetic data with {len(fonts)} fonts...")
    
    for digit in range(10):
        for i in range(num_samples_per_class):
            # Create blank image
            img_size = 96
            img = Image.new('L', (img_size, img_size), color=255)  # White background
            draw = ImageDraw.Draw(img)
            
            # Random font
            font = random.choice(fonts)
            
            # Random position offset
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-5, 5)
            
            # Draw digit
            text = str(digit)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img_size - text_width) // 2 + offset_x
            y = (img_size - text_height) // 2 + offset_y - 10
            
            draw.text((x, y), text, fill=0, font=font)  # Black text
            
            # Add random rotation
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
            
            # Add noise
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Determine split (80% train, 10% val, 10% test)
            if i < num_samples_per_class * 0.8:
                split = 'train'
            elif i < num_samples_per_class * 0.9:
                split = 'val'
            else:
                split = 'test'
            
            # Save image
            img_path = os.path.join(DATASET_PATH, split, 'images', f'digit_{digit}_{i:04d}.jpg')
            cv2.imwrite(img_path, cv2.cvtColor(np.stack([img_array]*3, axis=-1), cv2.COLOR_RGB2BGR))
            
            # Create label (full image as bounding box)
            label_path = os.path.join(DATASET_PATH, split, 'labels', f'digit_{digit}_{i:04d}.txt')
            with open(label_path, 'w') as f:
                f.write(f"{digit} 0.5 0.5 0.8 0.8\n")
    
    print(f"Generated {num_samples_per_class * 10} synthetic images!")


# ==================== TRAINING ====================

def train_yolov8_nano():
    """
    Train YOLOv8-nano model for digit detection.
    YOLOv8-nano is chosen for its small size suitable for ESP32.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    # Create dataset config
    yaml_path = create_yaml_config()
    
    # Load YOLOv8 nano model
    model = YOLO('yolov8n.pt')  # nano model - smallest
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='digit_detection',
        patience=20,
        save=True,
        device='cpu',  # Use 'cuda' if GPU available
        workers=2,
        project=MODEL_OUTPUT,
        exist_ok=True,
        # Augmentation
        augment=True,
        flipud=0.0,  # No vertical flip for digits
        fliplr=0.0,  # No horizontal flip for digits
        mosaic=0.5,
        mixup=0.1,
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: {MODEL_OUTPUT}/digit_detection/weights/best.pt")
    
    return model


def export_for_esp32(model_path):
    """
    Export trained model to TFLite format for ESP32.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return None
    
    model = YOLO(model_path)
    
    # Export to TFLite (INT8 quantized for ESP32)
    model.export(
        format='tflite',
        imgsz=IMG_SIZE,
        int8=True,  # INT8 quantization for smaller size
        data=os.path.join(DATASET_PATH, 'digit_dataset.yaml')
    )
    
    print("Model exported to TFLite format!")
    
    # Also export to ONNX for alternative deployment
    model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        simplify=True
    )
    
    print("Model also exported to ONNX format!")


# ==================== INFERENCE (PC Testing) ====================

def test_inference(model_path, test_image_path):
    """
    Test inference on PC before ESP32 deployment.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics")
        return
    
    model = YOLO(model_path)
    
    # Run inference
    results = model(test_image_path, imgsz=IMG_SIZE, conf=0.5)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"Detected digit: {class_id}, Confidence: {confidence:.2f}")
            print(f"Bounding box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
        
        # Save annotated image
        annotated = result.plot()
        cv2.imwrite('test_result.jpg', annotated)
        print("Result saved to test_result.jpg")


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EE 4065 - YOLO Handwritten Digit Detection Training")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    # Step 1: Generate/prepare dataset
    print("\n[Step 1] Preparing dataset...")
    generate_synthetic_data(num_samples_per_class=200)
    
    # Step 2: Train model
    print("\n[Step 2] Training YOLOv8-nano model...")
    model = train_yolov8_nano()
    
    # Step 3: Export for ESP32
    print("\n[Step 3] Exporting model for ESP32...")
    best_model_path = os.path.join(MODEL_OUTPUT, 'digit_detection', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        export_for_esp32(best_model_path)
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("Next steps:")
    print("1. Copy the TFLite model to ESP32-CAM")
    print("2. Use esp32_yolo_inference.ino for inference")
    print("=" * 60)


if __name__ == "__main__":
    main()
