"""
EE 4065 - Embedded Digital Image Processing
Question 5b (BONUS): SSD+MobileNet for Digit Detection

Student: KAAN ATALAY
ID: 150720057

SSD (Single Shot MultiBox Detector) with MobileNet backbone
for real-time object detection on ESP32-CAM.
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
NUM_CLASSES = 10
NUM_ANCHORS = 4  # Anchors per grid cell
EPOCHS = 50
BATCH_SIZE = 32
MODEL_DIR = "ssd_mobilenet_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== SSD ARCHITECTURE ====================

def create_mobilenet_backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """
    MobileNet backbone for SSD.
    Returns feature maps at multiple scales.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv + BN + ReLU6
    def conv_bn(x, filters, kernel_size, stride=1):
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    # Depthwise Separable Conv
    def dw_conv(x, filters, stride=1):
        x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    # Build backbone with multi-scale outputs
    x = conv_bn(inputs, 32, (3, 3), stride=2)  # 48x48
    x = dw_conv(x, 64)
    feat1 = dw_conv(x, 128, stride=2)  # 24x24 - Feature map 1
    
    x = dw_conv(feat1, 128)
    feat2 = dw_conv(x, 256, stride=2)  # 12x12 - Feature map 2
    
    x = dw_conv(feat2, 256)
    feat3 = dw_conv(x, 512, stride=2)  # 6x6 - Feature map 3
    
    x = dw_conv(feat3, 512)
    feat4 = dw_conv(x, 512, stride=2)  # 3x3 - Feature map 4
    
    return Model(inputs, [feat1, feat2, feat3, feat4], name='mobilenet_backbone')


def create_ssd_head(feature_map, num_anchors, num_classes, name_prefix):
    """
    SSD detection head for a single feature map.
    
    Outputs:
        - Box predictions: (batch, h, w, num_anchors * 4)
        - Class predictions: (batch, h, w, num_anchors * (num_classes + 1))
    """
    # Location prediction (4 values per anchor: dx, dy, dw, dh)
    loc = layers.Conv2D(num_anchors * 4, (3, 3), padding='same',
                        name=f'{name_prefix}_loc')(feature_map)
    
    # Confidence prediction (num_classes + 1 for background)
    conf = layers.Conv2D(num_anchors * (num_classes + 1), (3, 3), padding='same',
                         name=f'{name_prefix}_conf')(feature_map)
    
    return loc, conf


def create_ssd_mobilenet():
    """
    Complete SSD+MobileNet model.
    """
    backbone = create_mobilenet_backbone()
    inputs = backbone.input
    feat1, feat2, feat3, feat4 = backbone.outputs
    
    # Apply detection heads to each feature map
    loc1, conf1 = create_ssd_head(feat1, NUM_ANCHORS, NUM_CLASSES, 'head1')
    loc2, conf2 = create_ssd_head(feat2, NUM_ANCHORS, NUM_CLASSES, 'head2')
    loc3, conf3 = create_ssd_head(feat3, NUM_ANCHORS, NUM_CLASSES, 'head3')
    loc4, conf4 = create_ssd_head(feat4, NUM_ANCHORS, NUM_CLASSES, 'head4')
    
    # Reshape and concatenate predictions
    def reshape_output(x, name):
        batch_size = tf.shape(x)[0]
        return layers.Reshape((-1,), name=name)(x)
    
    # Flatten each output
    loc1_flat = reshape_output(loc1, 'loc1_flat')
    loc2_flat = reshape_output(loc2, 'loc2_flat')
    loc3_flat = reshape_output(loc3, 'loc3_flat')
    loc4_flat = reshape_output(loc4, 'loc4_flat')
    
    conf1_flat = reshape_output(conf1, 'conf1_flat')
    conf2_flat = reshape_output(conf2, 'conf2_flat')
    conf3_flat = reshape_output(conf3, 'conf3_flat')
    conf4_flat = reshape_output(conf4, 'conf4_flat')
    
    # Concatenate all predictions
    loc_output = layers.Concatenate(name='locations')([loc1_flat, loc2_flat, 
                                                        loc3_flat, loc4_flat])
    conf_output = layers.Concatenate(name='confidences')([conf1_flat, conf2_flat,
                                                          conf3_flat, conf4_flat])
    
    # Output layer
    outputs = layers.Concatenate(name='predictions')([loc_output, conf_output])
    
    return Model(inputs, outputs, name='ssd_mobilenet')


# ==================== ANCHOR GENERATION ====================

def generate_anchors():
    """
    Generate default anchor boxes for SSD.
    """
    feature_map_sizes = [24, 12, 6, 3]  # Sizes of feature maps
    scales = [0.1, 0.2, 0.4, 0.8]       # Anchor scales
    aspect_ratios = [1.0, 2.0, 0.5, 1.5]  # Aspect ratios
    
    anchors = []
    
    for fm_idx, fm_size in enumerate(feature_map_sizes):
        scale = scales[fm_idx]
        
        for i in range(fm_size):
            for j in range(fm_size):
                # Center of cell (normalized)
                cx = (j + 0.5) / fm_size
                cy = (i + 0.5) / fm_size
                
                for ar in aspect_ratios:
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)
                    
                    anchors.append([cx, cy, w, h])
    
    return np.array(anchors, dtype=np.float32)


# ==================== LOSS FUNCTIONS ====================

def ssd_loss(y_true, y_pred, num_classes=NUM_CLASSES, neg_pos_ratio=3.0):
    """
    SSD multi-task loss: localization + confidence.
    """
    # Split predictions
    num_anchors = len(generate_anchors())
    loc_pred = y_pred[:, :num_anchors * 4]
    conf_pred = y_pred[:, num_anchors * 4:]
    
    # Split ground truth
    loc_true = y_true[:, :num_anchors * 4]
    conf_true = y_true[:, num_anchors * 4:]
    
    # Reshape
    loc_pred = tf.reshape(loc_pred, [-1, num_anchors, 4])
    loc_true = tf.reshape(loc_true, [-1, num_anchors, 4])
    conf_pred = tf.reshape(conf_pred, [-1, num_anchors, num_classes + 1])
    conf_true = tf.reshape(conf_true, [-1, num_anchors, num_classes + 1])
    
    # Positive mask (non-background)
    pos_mask = tf.reduce_sum(conf_true[:, :, 1:], axis=-1) > 0
    num_pos = tf.maximum(tf.reduce_sum(tf.cast(pos_mask, tf.float32)), 1.0)
    
    # Localization loss (Smooth L1, only for positive anchors)
    loc_loss = tf.keras.losses.Huber()(loc_true, loc_pred)
    loc_loss = loc_loss * tf.cast(pos_mask, tf.float32)
    loc_loss = tf.reduce_sum(loc_loss) / num_pos
    
    # Confidence loss (Cross-entropy with hard negative mining)
    conf_loss = tf.keras.losses.categorical_crossentropy(conf_true, conf_pred)
    
    # Hard negative mining
    neg_mask = ~pos_mask
    neg_conf_loss = conf_loss * tf.cast(neg_mask, tf.float32)
    
    # Sort negatives and keep top-k
    num_neg = tf.minimum(num_pos * neg_pos_ratio, 
                         tf.reduce_sum(tf.cast(neg_mask, tf.float32)))
    neg_conf_loss_flat = tf.reshape(neg_conf_loss, [-1])
    _, neg_indices = tf.nn.top_k(neg_conf_loss_flat, k=tf.cast(num_neg, tf.int32))
    
    # Positive confidence loss
    pos_conf_loss = tf.reduce_sum(conf_loss * tf.cast(pos_mask, tf.float32)) / num_pos
    
    # Total loss
    total_loss = loc_loss + pos_conf_loss
    
    return total_loss


# ==================== DATA PREPARATION ====================

def encode_boxes(boxes, anchors):
    """
    Encode ground truth boxes relative to anchors.
    """
    encoded = np.zeros((len(anchors), 4), dtype=np.float32)
    
    for i, anchor in enumerate(anchors):
        # Find best matching box (simplified)
        ax, ay, aw, ah = anchor
        
        for box in boxes:
            bx, by, bw, bh = box[:4]
            
            # Encode as offset
            encoded[i, 0] = (bx - ax) / aw
            encoded[i, 1] = (by - ay) / ah
            encoded[i, 2] = np.log(bw / aw)
            encoded[i, 3] = np.log(bh / ah)
    
    return encoded


def generate_ssd_dataset(num_samples=5000):
    """Generate dataset for SSD training."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    anchors = generate_anchors()
    num_anchors = len(anchors)
    
    images = []
    labels = []
    
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 30)
    except:
        font = ImageFont.load_default()
    
    for _ in range(num_samples):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        
        boxes = []
        num_digits = random.randint(1, 2)
        
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            x = random.randint(5, IMG_SIZE - 25)
            y = random.randint(5, IMG_SIZE - 25)
            
            draw.text((x, y), str(digit), fill=0, font=font)
            
            # Box: [cx, cy, w, h, class]
            cx = (x + 10) / IMG_SIZE
            cy = (y + 15) / IMG_SIZE
            w = 20 / IMG_SIZE
            h = 25 / IMG_SIZE
            boxes.append([cx, cy, w, h, digit])
        
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, 8, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # RGB
        img_rgb = np.stack([img_array] * 3, axis=-1).astype(np.float32) / 255.0
        
        # Create label
        loc_label = encode_boxes(boxes, anchors)
        conf_label = np.zeros((num_anchors, NUM_CLASSES + 1), dtype=np.float32)
        conf_label[:, 0] = 1.0  # Background
        
        # Match boxes to anchors (simplified)
        for box in boxes:
            bx, by, bw, bh, class_id = box
            
            # Find best anchor
            best_idx = 0
            best_iou = 0
            
            for i, anchor in enumerate(anchors):
                ax, ay, aw, ah = anchor
                # Simple IoU approximation
                iou = max(0, min(bw, aw) * min(bh, ah) / (bw * bh + aw * ah))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > 0.3:
                conf_label[best_idx, 0] = 0  # Not background
                conf_label[best_idx, class_id + 1] = 1.0
        
        # Flatten labels
        label = np.concatenate([loc_label.flatten(), conf_label.flatten()])
        
        images.append(img_rgb)
        labels.append(label)
    
    return np.array(images), np.array(labels)


# ==================== TRAINING ====================

def train_ssd():
    """Train SSD+MobileNet model."""
    print("Generating dataset...")
    x_train, y_train = generate_ssd_dataset(5000)
    x_val, y_val = generate_ssd_dataset(1000)
    
    print(f"Training data: {x_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    
    # Create model
    model = create_ssd_mobilenet()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=ssd_loss
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'ssd_mobilenet_best.h5'),
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


def export_ssd_tflite(model):
    """Export SSD model to TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODEL_DIR, 'ssd_mobilenet_digit.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved SSD model: {len(tflite_model) / 1024:.1f} KB")


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EE 4065 - SSD+MobileNet Digit Detection Training")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    model, history = train_ssd()
    export_ssd_tflite(model)
    
    print("\nSSD+MobileNet training complete!")
    print("Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
