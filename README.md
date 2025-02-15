# Air Writing Digit Recognition with Deep Learning

This project involves creating an application that recognizes handwritten digits drawn in the air using hand gestures and deep learning. The model uses a Convolutional Neural Network (CNN) for digit recognition, and the hand gestures are captured through a webcam using MediaPipe.

## Features
- **Hand Gesture Recognition**: Using MediaPipe to detect hand movements and positions.
- **Digit Prediction**: Hand-drawn digits are processed and predicted using a pre-trained CNN model.
- **Real-time Interaction**: The user can draw digits in the air, and the model will predict the digit.

## Technologies Used
- **Python**: The primary programming language.
- **TensorFlow/Keras**: For building and training the CNN model.
- **MediaPipe**: For detecting and tracking hand gestures.
- **OpenCV**: For image processing and webcam integration.
  
## Requirements
To run the application, you will need the following Python libraries:

- TensorFlow
- Keras
- OpenCV
- NumPy
- Mediapipe

You can install the necessary dependencies by running:

```bash
pip install tensorflow opencv-python numpy mediapipe
