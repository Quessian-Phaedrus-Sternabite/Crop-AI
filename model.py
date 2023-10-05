import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
loaded_model = tf.keras.models.load_model('crop_segmentation_model.h5')

# Assuming you have a new image to predict on
new_image = cv2.imread('path/to/new_image.png')
new_image = cv2.resize(new_image, (256, 256))  # Resize to match model input shape
new_image = new_image / 255.0  # Normalize pixel values to [0, 1]

# Make a prediction using the loaded model
predicted_mask = loaded_model.predict(np.expand_dims(new_image, axis=0))

# The predicted_mask will contain the mask for healthy and unhealthy crops
import cv2
import numpy as np

# Assuming you have the predicted_mask as a NumPy array
# predicted_mask should be in the range [-1, 1], where values closer to 1 represent healthy crops
# and values closer to -1 represent unhealthy crops

# Normalize the predicted_mask to the range [0, 255] for saving as a PNG image
normalized_mask = ((predicted_mask + 1.0) / 2.0 * 255).astype(np.uint8)

# Save the mask as a PNG image
cv2.imwrite('predicted_mask.png', normalized_mask)