import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('moroccan_football_players_classifier.h5')

# Path to your test image
test_image_path = '/Users/oumaima/Downloads/62b6cbf831eb0_Zakaria-Aboukhlal (1)nnnnn.webp'

# Function to prepare image for prediction
def prepare_image(file):
    img = load_img(file, target_size=(150, 150))  # Resize the image to 150x150
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize the image
    return img_array

# Prepare the image
prepared_image = prepare_image(test_image_path)

# Make a prediction
prediction = model.predict(prepared_image)
predicted_class = np.argmax(prediction, axis=1)

# Print the predicted class
print(f'Predicted class for the image is: {predicted_class[0]}')
