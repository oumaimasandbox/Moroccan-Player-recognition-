# Moroccan Football Players Classifier

## Overview
This project is a deep learning model designed to classify images of Moroccan football players. It utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Dataset
The model is trained on a dataset of images of Moroccan football players. The data is augmented to improve the model's ability to generalize and to reduce overfitting.

## Model Architecture
The CNN model includes:
- Convolutional layers with ReLU activation.
- MaxPooling layers.
- A Flatten layer to convert the 2D features to 1D.
- A Dropout layer to reduce overfitting.
- Dense layers with ReLU and Softmax activations.

## Training
The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training is performed with data augmentation techniques like rotation, width and height shifting, shear mapping, zooming, horizontal flipping, and nearest fill mode.

## Performance
- Final Training Accuracy: {Final Training Accuracy}%
- Final Validation Accuracy: {Final Validation Accuracy}%

## Usage
To use the model:
1. Load the trained model.
2. Prepare the image for prediction by resizing and normalizing.
3. Make predictions and interpret the results.


## Example Prediction
 <img width="664" alt="Screenshot 2023-12-25 at 8 42 56 PM" src="https://github.com/oumaimasandbox/Moroccan-Player-recognition-/assets/77903484/3bc6b13c-2029-4a41-8692-bed2f32e52d4">
<img width="723" alt="Screenshot 2023-12-25 at 8 44 19 PM" src="https://github.com/oumaimasandbox/Moroccan-Player-recognition-/assets/77903484/11561738-449b-47c4-897b-caee591b4507">
