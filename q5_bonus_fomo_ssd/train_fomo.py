"""
EE 4065 - Embedded Digital Image Processing
Question 5a (BONUS): FOMO with Keras for Digit Detection

Student: KAAN ATALAY
ID: 150720057

FOMO (Faster Objects, More Objects) is a lightweight object detection model
designed for embedded systems. It provides bounding box centroids instead of
full boxes, making it extremely efficient.

Reference: https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import cv2

# ==================== CONFIGURATION ====================
IMG_SIZE = 96
GRID_SIZE = 12  # Output grid (96/8 = 12)
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 32
MODEL_DIR = "fomo_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== FOMO ARCHITECTURE ====================

def create_fomo_backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    FOMO Backbone: MobileNetV2-based feature extractor.
    Outputs feature map at 1/8 resolution.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv2D(16, (3, 3), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    # Depthwise separable blocks
    def dw_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    # Downsample to 1/8 (3 stride-2 layers: 96 -> 48 -> 24 -> 12)
    x = dw_block(x, 24, stride=2)  # 48x48
    x = dw_block(x, 32, stride=2)  # 24x24
    x = dw_block(x, 64, stride=2)  # 12x12
    
    # Additional feature extraction
    x = dw_block(x, 96, stride=1)
    x = dw_block(x, 128, stride=1)
    
    return Model(inputs, x, name='fomo_backbone')


def create_fomo_head(backbone, num_classes=NUM_CLASSES):
    """
    FOMO Detection Head: Predicts object presence and class per grid cell.
    
    Output: (batch, grid_h, grid_w, num_classes + 1)
    - Channel 0: Background/no-object
    - Channels 1-10: Class probabilities (digits 0-9)
    """
    inputs = backbone.input
    features = backbone.output
    
    # Detection head
    x = layers.Conv2D(64, (1, 1), padding='same')(features)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    x = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    # Final output: num_classes + 1 (background)
    outputs = layers.Conv2D(num_classes + 1, (1, 1), activation='softmax',
                           padding='same', name='fomo_output')(x)
    
    return Model(inputs, outputs, name='fomo_detector')


def create_fomo_model():
    """Create complete FOMO model."""
    backbone = create_fomo_backbone()
    model = create_fomo_head(backbone)
    return model


# ==================== DATA PREPARATION ====================

def create_fomo_labels(digit_positions, img_size=IMG_SIZE, grid_size=GRID_SIZE):
    """
    Create FOMO ground truth labels.
    
    Args:
        digit_positions: List of (x, y, class_id) tuples (normalized coordinates)
        
    Returns:
        Grid tensor of shape (grid_size, grid_size, num_classes + 1)
    """
    label = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
    
    # Initialize all cells as background
    label[:, :, 0] = 1.0
    
    for x, y, class_id in digit_positions:
        # Convert to grid coordinates
        grid_x = int(x * grid_size)
        grid_y = int(y * grid_size)
        
        # Clamp to valid range
        grid_x = min(max(grid_x, 0), grid_size - 1)
        grid_y = min(max(grid_y, 0), grid_size - 1)
        
        # Set class (class_id + 1 because 0 is background)
        label[grid_y, grid_x, 0] = 0.0  # Not background
        label[grid_y, grid_x, class_id + 1] = 1.0
    
    return label


def generate_fomo_dataset(num_samples=5000):
    """
    Generate synthetic dataset for FOMO training.
    Creates images with random digit placements.
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    images = []
    labels = []
    
    # Try to load a font
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 40)
    except:
        font = ImageFont.load_default()
    
    for _ in range(num_samples):
        # Create blank image
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        
        positions = []
        num_digits = random.randint(1, 3)  # 1-3 digits per image
        
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            
            # Random position (avoiding edges)
            x = random.randint(10, IMG_SIZE - 30)
            y = random.randint(10, IMG_SIZE - 30)
            
            # Draw digit
            draw.text((x, y), str(digit), fill=0, font=font)
            
            # Store normalized position (center)
            cx = (x + 15) / IMG_SIZE
            cy = (y + 20) / IMG_SIZE
            positions.append((cx, cy, digit))
        
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = np.stack([img_array] * 3, axis=-1).astype(np.float32) / 255.0
        
        # Create label
        label = create_fomo_labels(positions)
        
        images.append(img_rgb)
        labels.append(label)
    
    return np.array(images), np.array(labels)


# ==================== LOSS FUNCTION ====================

def fomo_loss(y_true, y_pred):
    """
    FOMO loss: Focal cross-entropy for handling class imbalance.
    """
    # Focal loss parameters
    gamma = 2.0
    alpha = 0.25
    
    # Clip predictions
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Cross entropy
    ce = -y_true * tf.math.log(y_pred)
    
    # Focal weight
    focal_weight = alpha * tf.pow(1 - y_pred, gamma)
    
    # Apply weights
    loss = focal_weight * ce
    
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# ==================== TRAINING ====================

def train_fomo():
    """Train FOMO model."""
    print("Generating dataset...")
    x_train, y_train = generate_fomo_dataset(5000)
    x_val, y_val = generate_fomo_dataset(1000)
    
    print(f"Training data: {x_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    
    # Create model
    model = create_fomo_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=fomo_loss,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'fomo_best.h5'),
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    return model, history


def export_fomo_tflite(model):
    """Export FOMO model to TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODEL_DIR, 'fomo_digit.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved FOMO model: {len(tflite_model) / 1024:.1f} KB")


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EE 4065 - FOMO Digit Detection Training")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    model, history = train_fomo()
    export_fomo_tflite(model)
    
    print("\nFOMO training complete!")
    print("Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
