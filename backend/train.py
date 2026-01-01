"""
train.py - Run this ONCE to train your MobileNetV2 model on the fruits dataset

Usage:
    python train.py

Output:
    - mobilenet_fruits_model.h5  (trained model)
    - class_labels.json          (fruit class names)
"""

import kagglehub
import numpy as np
import os
import json
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 60)
print("TRAINING MOBILENETV2 FOR FRUIT CLASSIFICATION")
print("=" * 60)

# Download dataset
print("\n[1/6] Downloading Kaggle dataset...")
dataset_path = kagglehub.dataset_download("moltean/fruits")
print(f"✓ Dataset downloaded to: {dataset_path}")

# Setup paths
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Test')

if not os.path.exists(train_dir):
    print(f"ERROR: Training directory not found at {train_dir}")
    print("Please check the dataset structure")
    exit(1)

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators with augmentation
print("\n[2/6] Setting up data generators...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"✓ Found {num_classes} fruit classes")
print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {val_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")

# Create model
print("\n[3/6] Building model architecture...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture created")
print(f"  Total params: {model.count_params():,}")

# Callbacks
checkpoint = ModelCheckpoint(
    'mobilenet_fruits_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Phase 1: Train top layers
print("\n[4/6] Phase 1: Training classifier layers (10 epochs)...")
print("-" * 60)
history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

print(f"\n✓ Phase 1 complete")
print(f"  Best validation accuracy: {max(history1.history['val_accuracy']):.2%}")

# Phase 2: Fine-tune
print("\n[5/6] Phase 2: Fine-tuning last 20 layers (10 epochs)...")
print("-" * 60)
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

print(f"\n✓ Phase 2 complete")
print(f"  Best validation accuracy: {max(history2.history['val_accuracy']):.2%}")

# Evaluate
print("\n[6/6] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"✓ Test accuracy: {test_accuracy:.2%}")

# Save final model
model_path = "mobilenet_fruits_model.h5"
model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
labels_path = "class_labels.json"
with open(labels_path, 'w') as f:
    json.dump(class_labels, f, indent=2)
print(f"✓ Class labels saved to: {labels_path}")

# Summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model file: {model_path}")
print(f"Labels file: {labels_path}")
print(f"Number of classes: {num_classes}")
print(f"Test accuracy: {test_accuracy:.2%}")
print("\nYou can now run your FastAPI app with:")
print("  uvicorn main:app --reload")
print("=" * 60)