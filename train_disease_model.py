# train_disease_model.py - Coconut Disease CNN Training (Normal Python File)

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import classification_report

# -------------------------------
# Project base path
# -------------------------------
BASE_DIR = r"D:\coconut_project"

# Correct DATA_DIR – double path இல்லாம
DATA_DIR = os.path.join(BASE_DIR, "data", "disease_images")

print("Dataset path:", DATA_DIR)

# Path check
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Folder not found: {DATA_DIR}")
    try:
        print("Available folders in 'data':", os.listdir(os.path.join(BASE_DIR, "data")))
    except Exception as e:
        print("Even 'data' folder not found:", str(e))
    exit(1)

print("Classes found:", os.listdir(DATA_DIR))

# -------------------------------
# Classes (உன் folder names exact ஆக match பண்ணு)
# -------------------------------
classes = [
    'Bud Rot', 'Bud Root Dropping', 'CCI_Caterpillars', 'CCI_Leaflets',
    'Gray Leaf Spot', 'Healthy Leaves', 'Leaf Rot', 'Stem Bleeding',
    'WCLWD_DryingofLeaflets', 'WCLWD_Flaccidity', 'WCLWD_Yellowing'
]

# -------------------------------
# Data generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=classes,
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=classes,
    shuffle=False
)

# Class info
class_indices = train_generator.class_indices
num_classes = len(class_indices)
print("\nClass indices:", class_indices)
print("Number of classes:", num_classes)

# -------------------------------
# Build CNN model
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# Train
# -------------------------------
print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=20,  
    validation_data=val_generator,
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
model_save_path = os.path.join(BASE_DIR, 'models', 'coconut_disease_cnn.h5')
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Save class names for app.py
import json
with open(os.path.join(BASE_DIR, 'models', 'disease_classes.json'), 'w') as f:
    json.dump(list(class_indices.keys()), f)
print("Class names saved to: models/disease_classes.json")