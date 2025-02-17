# -*- coding: utf-8 -*-
"""Train_Model.ipynb"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define INPUT_SIZE at the top
INPUT_SIZE = 224

# Function to load and preprocess images
def load_and_preprocess_image(image_path, input_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = preprocess_input(image)
    return image

# Load images and labels
def load_data(image_dir):
    dataset = []
    labels = []
    label_map = {'No': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate': 4}
    for label_name in label_map.keys():
        image_paths = os.listdir(os.path.join(image_dir, label_name))
        for image_name in image_paths:
            # Check for multiple image formats
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, label_name, image_name)
                image = load_and_preprocess_image(image_path, INPUT_SIZE)
                dataset.append(image)
                labels.append(label_map[label_name])
    return np.array(dataset), np.array(labels)

# Load data
image_directory = 'dataset/'

dataset, labels = load_data(image_directory)

# Split data into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Load pre-trained DenseNet121 model (experiment with different architectures)
base_model = DenseNet121(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False, weights='imagenet')

# Freeze some base model layers (adjust number of layers to unfreeze for fine-tuning)
base_model.trainable = False  # Freeze all layers by default

# Optionally, select layers to unfreeze for fine-tuning
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers for fine-tuning
    layer.trainable = True

# Add custom classification head
global_average_layer = layers.GlobalAveragePooling2D()
dropout = layers.Dropout(0.5)  # Add dropout for regularization
output_layer = layers.Dense(5, activation='softmax')

# Create the model
model = models.Sequential([
  base_model,
  global_average_layer,
  dropout,
  output_layer
])

# Compile the model (change 'lr' to 'learning_rate')
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),  # Adjust the learning rate as needed
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model with data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Train the model
history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=15,
                    validation_data=val_datagen.flow(val_images, val_labels, batch_size=32),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Save the model after training in Keras format
model.save('heart_attack_prediction_model.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# List all data in history
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predictions with batch size
predictions = model.predict(test_images, batch_size=32)
predicted_labels = np.argmax(predictions, axis=1)

# Create a confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Mild', 'Moderate', 'Severe', 'Proliferate'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
