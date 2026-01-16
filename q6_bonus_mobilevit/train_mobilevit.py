"""
EE 4065 - Embedded Digital Image Processing
Question 6 (BONUS): MobileViT for Digit Detection

Student: KAAN ATALAY
ID: 150720057

MobileViT combines the strengths of CNNs and Vision Transformers
for efficient mobile vision applications.

Reference: https://keras.io/examples/vision/mobilevit/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import cv2

# ==================== CONFIGURATION ====================
IMG_SIZE = 96
PATCH_SIZE = 4
NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 32
MODEL_DIR = "mobilevit_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== MOBILEVIT COMPONENTS ====================

def conv_block(x, filters, kernel_size=3, strides=1, name=None):
    """Standard convolution block with BN and activation."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, 
                      padding='same', use_bias=False, name=f'{name}_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('swish', name=f'{name}_swish')(x)
    return x


def inverted_residual_block(x, expanded_channels, output_channels, strides=1, name=None):
    """MobileNetV2-style inverted residual block."""
    input_channels = x.shape[-1]
    
    # Expansion
    m = conv_block(x, expanded_channels, kernel_size=1, name=f'{name}_expand')
    
    # Depthwise
    m = layers.DepthwiseConv2D(3, strides=strides, padding='same', 
                                use_bias=False, name=f'{name}_dw')(m)
    m = layers.BatchNormalization(name=f'{name}_dw_bn')(m)
    m = layers.Activation('swish', name=f'{name}_dw_swish')(m)
    
    # Projection
    m = layers.Conv2D(output_channels, 1, padding='same', 
                      use_bias=False, name=f'{name}_project')(m)
    m = layers.BatchNormalization(name=f'{name}_project_bn')(m)
    
    # Residual connection
    if strides == 1 and input_channels == output_channels:
        return layers.Add(name=f'{name}_add')([x, m])
    return m


def mlp(x, hidden_units, dropout_rate, name=None):
    """Multi-layer perceptron for transformer."""
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation='swish', name=f'{name}_dense{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'{name}_dropout{i}')(x)
    return x


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self attention layer."""
    
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.combine = layers.Dense(embed_dim)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_probs, v)
        
        # Reshape back
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, seq_len, self.embed_dim])
        
        return self.combine(attention_output)


def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1, name=None):
    """Transformer encoder block."""
    # Layer normalization 1
    x1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(x)
    
    # Multi-head attention
    attention_output = MultiHeadSelfAttention(embed_dim, num_heads, 
                                               name=f'{name}_mhsa')(x1)
    attention_output = layers.Dropout(dropout, name=f'{name}_attn_dropout')(attention_output)
    
    # Skip connection 1
    x2 = layers.Add(name=f'{name}_add1')([x, attention_output])
    
    # Layer normalization 2
    x3 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x2)
    
    # MLP
    mlp_output = mlp(x3, [mlp_dim, embed_dim], dropout, name=f'{name}_mlp')
    
    # Skip connection 2
    return layers.Add(name=f'{name}_add2')([x2, mlp_output])


def mobilevit_block(x, num_blocks, projection_dim, patch_size=PATCH_SIZE, name=None):
    """
    MobileViT block: unfold -> transformer -> fold.
    Combines local and global information processing.
    """
    local_features = x
    
    # Local representation (conv)
    local_features = conv_block(local_features, projection_dim, kernel_size=3, 
                                name=f'{name}_local_rep')
    
    # Get spatial dimensions
    batch_size = tf.shape(local_features)[0]
    h = tf.shape(local_features)[1]
    w = tf.shape(local_features)[2]
    channels = local_features.shape[-1]
    
    # Ensure dimensions are compatible with patch size
    # Unfold to patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    # Reshape to patches: (batch, num_patches, patch_size*patch_size, channels)
    patches = tf.reshape(local_features, 
                         [batch_size, num_patches_h, patch_size, num_patches_w, patch_size, channels])
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    patches = tf.reshape(patches, [batch_size, num_patches_h * num_patches_w, 
                                    patch_size * patch_size, channels])
    
    # Flatten patches for transformer
    patches = tf.reshape(patches, [batch_size, -1, channels])
    
    # Apply transformer blocks
    for i in range(num_blocks):
        patches = transformer_block(patches, projection_dim, num_heads=4,
                                   mlp_dim=projection_dim * 2, name=f'{name}_transformer{i}')
    
    # Fold back to feature map
    patches = tf.reshape(patches, [batch_size, num_patches_h, num_patches_w, 
                                   patch_size, patch_size, channels])
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    global_features = tf.reshape(patches, [batch_size, h, w, channels])
    
    # Project and combine
    global_features = conv_block(global_features, projection_dim, kernel_size=1,
                                 name=f'{name}_global_proj')
    
    # Concatenate local and global features
    combined = layers.Concatenate(name=f'{name}_concat')([x, global_features])
    combined = conv_block(combined, projection_dim, kernel_size=3, name=f'{name}_fusion')
    
    return combined


def create_mobilevit(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Create MobileViT model optimized for ESP32.
    Smaller version suitable for embedded deployment.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv
    x = conv_block(inputs, 16, strides=2, name='initial')  # 48x48
    
    # MobileNetV2 blocks
    x = inverted_residual_block(x, 32, 24, strides=2, name='ir1')  # 24x24
    x = inverted_residual_block(x, 48, 24, strides=1, name='ir2')
    
    x = inverted_residual_block(x, 72, 48, strides=2, name='ir3')  # 12x12
    x = inverted_residual_block(x, 96, 48, strides=1, name='ir4')
    
    # MobileViT block 1
    x = mobilevit_block(x, num_blocks=2, projection_dim=64, name='mvit1')
    
    # More inverted residual blocks
    x = inverted_residual_block(x, 128, 64, strides=2, name='ir5')  # 6x6
    
    # MobileViT block 2
    x = mobilevit_block(x, num_blocks=2, projection_dim=80, patch_size=2, name='mvit2')
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    return Model(inputs, outputs, name='MobileViT_Lite')


# ==================== DATA PREPARATION ====================

def prepare_data():
    """Prepare MNIST data for MobileViT."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Resize to target size
    x_train = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in x_train])
    x_test = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in x_test])
    
    # Convert to RGB (3 channels)
    x_train = np.stack([x_train] * 3, axis=-1).astype('float32') / 255.0
    x_test = np.stack([x_test] * 3, axis=-1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    return (x_train, y_train), (x_test, y_test)


# ==================== TRAINING ====================

def train_mobilevit():
    """Train MobileViT model."""
    print("Preparing data...")
    (x_train, y_train), (x_test, y_test) = prepare_data()
    
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    # Create model
    model = create_mobilevit()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'mobilevit_best.h5'),
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Evaluate
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    return model, history


def export_to_tflite(model):
    """Export MobileViT to TFLite with quantization."""
    # Save Keras model
    model.save(os.path.join(MODEL_DIR, 'mobilevit.h5'))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Full integer quantization for ESP32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Representative dataset for calibration
    def representative_dataset():
        (x_train, _), _ = prepare_data()
        for i in range(100):
            yield [x_train[i:i+1]]
    
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save
    tflite_path = os.path.join(MODEL_DIR, 'mobilevit_digit.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved MobileViT model: {len(tflite_model) / 1024:.1f} KB")
    
    # Convert to C header
    os.system(f"xxd -i {tflite_path} > {os.path.join(MODEL_DIR, 'mobilevit_model.h')}")


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EE 4065 - MobileViT Digit Detection Training")
    print("Student: KAAN ATALAY (150720057)")
    print("=" * 60)
    
    model, history = train_mobilevit()
    export_to_tflite(model)
    
    print("\nMobileViT training complete!")
    print("Model saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
