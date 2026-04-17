import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- SETTINGS ---
IMG_SIZE = (64, 64)
N_CLASSES = 6
CLASS_NAMES = ['gesture1', 'gesture2', 'gesture3', 'gesture4', 'gesture5', 'unknown']
REAL_DATA_PATH = '/home/surya/Desktop/own/processed/realdata'
SYN_DATA_PATH = '/home/surya/Desktop/own/processed/gestures_synthetic'
SAVE_MODEL_PATH = '/home/surya/Desktop/4/model/Xgesture_cnn_dw_bn.h5'
EPOCHS = 20
BATCH_SIZE = 32

def load_data_from_folder(folder_path):
    X, y = [], []
    for idx, gesture in enumerate(CLASS_NAMES):
        folder = os.path.join(folder_path, gesture)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None or img.shape[:2] != IMG_SIZE:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

# Load real and synthetic data
X_real, y_real = load_data_from_folder(REAL_DATA_PATH)
X_syn, y_syn = load_data_from_folder(SYN_DATA_PATH)

# Combine both datasets
X = np.concatenate([X_real, X_syn], axis=0)
y = np.concatenate([y_real, y_syn], axis=0)

print(f"Total samples: {len(X)}, Labels: {np.bincount(y)}")

# Normalize data
X = X.astype('float32') 

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# --- MODEL DEFINITION (MobileNet-style, with softmax for training) ---
def get_pynq_hgr_model_v2(input_shape=(64,64,3), num_classes=6):
    x = inputs = layers.Input(input_shape)
    # Block 1
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(8, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)   # 32x32x8
    # Block 2
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)   # 16x16x16
    # Block 3
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)   # 8x8x32
    # Block 4
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)   # 4x4x32
    # Classifier
    x = layers.Flatten()(x)         # 512
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)  # <--- Softmax for training
    return models.Model(inputs, x)

model = get_pynq_hgr_model_v2(input_shape=(*IMG_SIZE, 3), num_classes=N_CLASSES)
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save model (for quantization/export)
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
model.save(SAVE_MODEL_PATH)
print(f"✅ Model saved to {SAVE_MODEL_PATH}")

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.3f}")
