"""
EE 4065 - Embedded Digital Image Processing
Question 4: Multi-Model Handwritten Digit Recognition

Student: KAAN ATALAY
ID: 150720057

This script trains multiple models (SqueezeNet, EfficientNet, MobileNet, ResNet)
for handwritten digit recognition and prepares them for ESP32-CAM deployment.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2

# ==================== CONFIGURATION ====================
IMG_SIZE = 96  # Input size for ESP32 compatibility
NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = "trained_models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== DATA PREPARATION ====================

def prepare_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Resize to target size and add channel dimension
    x_train_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in x_test])
    
    # Convert to 3 channels (RGB) for compatibility with pretrained models
    x_train_rgb = np.stack([x_train_resized] * 3, axis=-1)
    x_test_rgb = np.stack([x_test_resized] * 3, axis=-1)
    
    # Normalize to [0, 1]
    x_train_rgb = x_train_rgb.astype('float32') / 255.0
    x_test_rgb = x_test_rgb.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    print(f"Training data shape: {x_train_rgb.shape}")
    print(f"Test data shape: {x_test_rgb.shape}")
    
    return (x_train_rgb, y_train_cat), (x_test_rgb, y_test_cat)


# ==================== MODEL ARCHITECTURES ====================

def create_squeezenet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    SqueezeNet implementation optimized for ESP32.
    Uses Fire modules for efficient feature extraction.
    """
    
    def fire_module(x, squeeze_filters, expand_filters, name):
        """Fire module: squeeze then expand."""
        # Squeeze layer
        squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu',
                               padding='same', name=f'{name}_squeeze')(x)
        
        # Expand layers
        expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu',
                                   padding='same', name=f'{name}_expand_1x1')(squeeze)
        expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu',
                                   padding='same', name=f'{name}_expand_3x3')(squeeze)
        
        # Concatenate
        return layers.Concatenate(name=f'{name}_concat')([expand_1x1, expand_3x3])
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Fire modules
    x = fire_module(x, 16, 64, 'fire2')
    x = fire_module(x, 16, 64, 'fire3')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = fire_module(x, 32, 128, 'fire4')
    x = fire_module(x, 32, 128, 'fire5')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = fire_module(x, 48, 192, 'fire6')
    x = fire_module(x, 48, 192, 'fire7')
    
    # Final layers
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation('softmax')(x)
    
    model = Model(inputs, outputs, name='SqueezeNet')
    return model


def create_mobilenet_v2(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    MobileNetV2-inspired model optimized for ESP32.
    Uses depthwise separable convolutions and inverted residuals.
    """
    
    def inverted_residual_block(x, filters, stride, expansion, name):
        """Inverted residual block with depthwise separable conv."""
        in_channels = x.shape[-1]
        expanded_channels = in_channels * expansion
        
        # Expansion
        if expansion != 1:
            x_expanded = layers.Conv2D(expanded_channels, (1, 1), padding='same',
                                       name=f'{name}_expand')(x)
            x_expanded = layers.BatchNormalization(name=f'{name}_expand_bn')(x_expanded)
            x_expanded = layers.ReLU(6.0, name=f'{name}_expand_relu')(x_expanded)
        else:
            x_expanded = x
        
        # Depthwise
        x_dw = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same',
                                       name=f'{name}_depthwise')(x_expanded)
        x_dw = layers.BatchNormalization(name=f'{name}_dw_bn')(x_dw)
        x_dw = layers.ReLU(6.0, name=f'{name}_dw_relu')(x_dw)
        
        # Projection
        x_proj = layers.Conv2D(filters, (1, 1), padding='same',
                               name=f'{name}_project')(x_dw)
        x_proj = layers.BatchNormalization(name=f'{name}_project_bn')(x_proj)
        
        # Residual connection
        if stride == 1 and in_channels == filters:
            return layers.Add(name=f'{name}_add')([x, x_proj])
        return x_proj
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    
    # Inverted residual blocks
    x = inverted_residual_block(x, 16, 1, 1, 'block1')
    x = inverted_residual_block(x, 24, 2, 6, 'block2')
    x = inverted_residual_block(x, 24, 1, 6, 'block3')
    x = inverted_residual_block(x, 32, 2, 6, 'block4')
    x = inverted_residual_block(x, 32, 1, 6, 'block5')
    x = inverted_residual_block(x, 64, 2, 6, 'block6')
    x = inverted_residual_block(x, 64, 1, 6, 'block7')
    x = inverted_residual_block(x, 96, 1, 6, 'block8')
    
    # Final layers
    x = layers.Conv2D(256, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='MobileNetV2_Lite')
    return model


def create_efficientnet_lite(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    EfficientNet-Lite inspired model for ESP32.
    Simplified version with reduced complexity.
    """
    
    def mb_conv_block(x, filters, kernel_size, stride, expansion, se_ratio, name):
        """Mobile inverted bottleneck convolution block."""
        in_channels = x.shape[-1]
        expanded_channels = in_channels * expansion
        
        # Expansion phase
        if expansion != 1:
            x_exp = layers.Conv2D(expanded_channels, (1, 1), padding='same',
                                  name=f'{name}_expand_conv')(x)
            x_exp = layers.BatchNormalization(name=f'{name}_expand_bn')(x_exp)
            x_exp = layers.Activation('swish', name=f'{name}_expand_activation')(x_exp)
        else:
            x_exp = x
        
        # Depthwise convolution
        x_dw = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same',
                                       name=f'{name}_dw_conv')(x_exp)
        x_dw = layers.BatchNormalization(name=f'{name}_dw_bn')(x_dw)
        x_dw = layers.Activation('swish', name=f'{name}_dw_activation')(x_dw)
        
        # Squeeze and Excitation (simplified)
        if se_ratio > 0:
            se_filters = max(1, int(in_channels * se_ratio))
            se = layers.GlobalAveragePooling2D(name=f'{name}_se_squeeze')(x_dw)
            se = layers.Reshape((1, 1, expanded_channels), name=f'{name}_se_reshape')(se)
            se = layers.Conv2D(se_filters, (1, 1), activation='swish',
                              padding='same', name=f'{name}_se_reduce')(se)
            se = layers.Conv2D(expanded_channels, (1, 1), activation='sigmoid',
                              padding='same', name=f'{name}_se_expand')(se)
            x_dw = layers.Multiply(name=f'{name}_se_excite')([x_dw, se])
        
        # Output projection
        x_proj = layers.Conv2D(filters, (1, 1), padding='same',
                               name=f'{name}_project_conv')(x_dw)
        x_proj = layers.BatchNormalization(name=f'{name}_project_bn')(x_proj)
        
        # Skip connection
        if stride == 1 and in_channels == filters:
            x_proj = layers.Add(name=f'{name}_add')([x, x_proj])
        
        return x_proj
    
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_activation')(x)
    
    # Blocks
    x = mb_conv_block(x, 16, (3, 3), 1, 1, 0.25, 'block1')
    x = mb_conv_block(x, 24, (3, 3), 2, 6, 0.25, 'block2')
    x = mb_conv_block(x, 24, (3, 3), 1, 6, 0.25, 'block3')
    x = mb_conv_block(x, 40, (5, 5), 2, 6, 0.25, 'block4')
    x = mb_conv_block(x, 40, (5, 5), 1, 6, 0.25, 'block5')
    x = mb_conv_block(x, 80, (3, 3), 2, 6, 0.25, 'block6')
    x = mb_conv_block(x, 80, (3, 3), 1, 6, 0.25, 'block7')
    
    # Head
    x = layers.Conv2D(128, (1, 1), padding='same', name='head_conv')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Activation('swish', name='head_activation')(x)
    x = layers.GlobalAveragePooling2D(name='head_gap')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='EfficientNet_Lite')
    return model


def create_resnet_lite(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    ResNet-inspired lightweight model for ESP32.
    Uses fewer filters and simpler residual blocks.
    """
    
    def residual_block(x, filters, stride=1, name=''):
        """Basic residual block."""
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                         name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.ReLU(name=f'{name}_relu1')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # Shortcut path
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                     padding='same', name=f'{name}_shortcut')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
        
        # Add shortcut
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name}_out_relu')(x)
        
        return x
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
    
    # Residual blocks
    x = residual_block(x, 32, 1, 'block1_1')
    x = residual_block(x, 32, 1, 'block1_2')
    
    x = residual_block(x, 64, 2, 'block2_1')
    x = residual_block(x, 64, 1, 'block2_2')
    
    x = residual_block(x, 128, 2, 'block3_1')
    x = residual_block(x, 128, 1, 'block3_2')
    
    # Final layers
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='ResNet_Lite')
    return model


# ==================== TRAINING ====================

def train_model(model, train_data, test_data, epochs=EPOCHS):
    """Train a model and return history."""
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def export_to_tflite(model, model_name):
    """Convert model to TFLite with INT8 quantization."""
    # Save Keras model
    model.save(os.path.join(MODEL_DIR, f'{model_name}.h5'))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(MODEL_DIR, f'{model_name}.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved {model_name}.tflite ({len(tflite_model) / 1024:.1f} KB)")
    
    return tflite_path


# ==================== ENSEMBLE FUSION ====================

def ensemble_predict(models, x):
    """
    Ensemble prediction using multiple models.
    Combines predictions using weighted averaging.
    """
    predictions = []
    weights = []
    
    for model in models:
        pred = model.predict(x, verbose=0)
        predictions.append(pred)
        # Weight based on model accuracy (simplified)
        weights.append(1.0 / len(models))
    
    # Weighted average
    weighted_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        weighted_pred += pred * weight
    
    return weighted_pred


def majority_voting(models, x):
    """
    Ensemble prediction using majority voting.
    """
    votes = np.zeros((len(x), NUM_CLASSES))
    
    for model in models:
        pred = model.predict(x, verbose=0)
        class_pred = np.argmax(pred, axis=1)
        for i, c in enumerate(class_pred):
            votes[i, c] += 1
    
    return np.argmax(votes, axis=1)


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EE 4065 - Multi-Model Digit Recognition Training")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    # Prepare data
    print("\n[Step 1] Preparing data...")
    train_data, test_data = prepare_data()
    
    # Create models
    print("\n[Step 2] Creating models...")
    models = {
        'squeezenet': create_squeezenet(),
        'mobilenet_v2': create_mobilenet_v2(),
        'efficientnet_lite': create_efficientnet_lite(),
        'resnet_lite': create_resnet_lite()
    }
    
    # Print model summaries
    for name, model in models.items():
        print(f"\n{name.upper()} Summary:")
        print(f"  Parameters: {model.count_params():,}")
    
    # Train each model
    print("\n[Step 3] Training models...")
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        print(f"{'='*40}")
        
        history = train_model(model, train_data, test_data)
        trained_models[name] = model
        
        # Evaluate
        _, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
        results[name] = accuracy
        print(f"{name} Test Accuracy: {accuracy*100:.2f}%")
        
        # Export to TFLite
        export_to_tflite(model, name)
    
    # Ensemble evaluation
    print("\n[Step 4] Ensemble Evaluation...")
    
    # Weighted average ensemble
    x_test, y_test = test_data
    ensemble_pred = ensemble_predict(list(trained_models.values()), x_test)
    ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Ensemble (Weighted Average) Accuracy: {ensemble_acc*100:.2f}%")
    
    # Majority voting
    voting_pred = majority_voting(list(trained_models.values()), x_test)
    voting_acc = np.mean(voting_pred == np.argmax(y_test, axis=1))
    print(f"Ensemble (Majority Voting) Accuracy: {voting_acc*100:.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:20s}: {acc*100:.2f}%")
    print(f"  {'Ensemble (Avg)':20s}: {ensemble_acc*100:.2f}%")
    print(f"  {'Ensemble (Vote)':20s}: {voting_acc*100:.2f}%")
    print("=" * 60)
    
    print("\nModels saved to:", MODEL_DIR)
    print("Ready for ESP32-CAM deployment!")


if __name__ == "__main__":
    main()
