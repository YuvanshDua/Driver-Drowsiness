import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random, shutil

def generator(dir, gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

BS = 32  # Batch size
TS = (24, 24)  # Target size for images

# Generator setup
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# Calculate steps per epoch and validation steps
SPE = train_batch.samples // BS  # Total number of training samples divided by batch size
VS = valid_batch.samples // BS  # Total number of validation samples divided by batch size

# Print to check the values
print(f'STEPS_PER_EPOCH: {SPE}, VALIDATION_STEPS: {VS}')

# Check the shape of one batch of images
img, labels = next(train_batch)
print(img.shape)

# Model definition using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Save the trained model
model.save('models/cnnCat2.h5', overwrite=True)
