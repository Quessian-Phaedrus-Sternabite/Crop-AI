# Import necessary libraries
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.optimizers import Adam 
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time

print(f"import complete {time.time}")

# Define the U-Net model
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Add more convolutional and pooling layers as needed
    # Uneeded for testing

    # Decoder
    up2 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(pool1))
    concat2 = Concatenate()([conv1, up2])
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(concat2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)

    # Add more upsampling and convolutional layers as needed

    # Output layer
    output = Conv2D(1, 1, activation='tanh', padding='same')(conv2)  # Output in the range [-1, 1]

    model = Model(inputs=inputs, outputs=output)
    return model

# Create the model
model = unet_model(input_shape=(256, 256, 3))  # Adjust input_shape as per your image size

print(f"model created {time.time}")

# compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

print(f"model compiled {time.time}")

# Image recognition
# Define directories for training, validation, and testing data
train_image_dir = 'path/to/train/images'
train_mask_dir = 'path/to/train/masks'
validation_image_dir = 'path/to/validation/images'
validation_mask_dir = 'path/to/validation/masks'
test_image_dir = 'path/to/test/images'
test_mask_dir = 'path/to/test/masks'

# Initialize lists to store data
train_images = []
train_masks = []
val_images = []
val_masks = []
test_images = []
test_masks = []

# Function to load and preprocess data
def load_and_preprocess_data(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):  # Adjust the file format as needed
            # Load and preprocess the image
            img = cv2.imread(os.path.join(image_dir, filename))
            img = cv2.resize(img, (256, 256))  # Resize to match model input shape
            img = img / 255.0  # Normalize pixel values to [0, 1]
            
            # Load and preprocess the corresponding mask
            mask_filename = filename[:-4] + "_mask.png"  # Assuming mask filenames are related to image filenames
            mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))  # Resize to match model input shape
            mask = (mask / 255.0) * 2.0 - 1.0  # Normalize pixel values to [-1, 1]

            images.append(img)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load and preprocess training data
train_images, train_masks = load_and_preprocess_data(train_image_dir, train_mask_dir)

# Split training data into training and validation sets
val_split = 0.15  # Adjust the validation split ratio
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=val_split, random_state=42)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(validation_image_dir, validation_mask_dir)

# Load and preprocess testing data
test_images, test_masks = load_and_preprocess_data(test_image_dir, test_mask_dir)

print(f"images processed {time.time}")

# Define training parameters
num_epochs = 10 
batch_size = 32

# Add in crop data
model.fit(x=train_images, y=train_masks, validation_data=(val_images, val_masks), epochs=num_epochs, batch_size=batch_size)

print(f"Training complete {time.time}")

# Save the trained model to a file
model.save('crop_segmentation_model_1.h5')
