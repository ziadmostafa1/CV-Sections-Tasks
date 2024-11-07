# Sign Language Recognition

This project is a real-time sign language recognition application that uses MediaPipe for hand landmark extraction, TensorFlow for training a neural network model, and Streamlit for creating an interactive web interface.

## Project Overview

The application recognizes 45 different sign language gestures, including letters, numbers, and commonly used phrases:
1. Letters (A-Z)
2. Numbers (0-9)
3. Common phrases like "Hello," "Thank You," and "Excuse Me."

Users can either upload an image or use a webcam feed to classify the sign language gesture, with the model displaying predictions and confidence scores.

![](demo_split-1.gif)

## Table of Contents

- [Sign Language Recognition](#sign-language-recognition)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Technologies Used](#technologies-used)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Model Training](#model-training)
    - [To Train the Model](#to-train-the-model)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)

## Technologies Used

- **Python**
- **TensorFlow**: for model training and inference.
- **MediaPipe**: for hand landmark extraction.
- **Streamlit**: for creating an interactive user interface.
- **OpenCV**: for webcam and image processing.

## Setup

1. **Clone this repository**:
   ```bash
   # Initialize git repo
   git init Hands-Sign
   cd Hands-Sign

   # Add remote and enable sparse checkout
   git remote add origin https://github.com/ziadmostafa1/CV-Sections-Tasks.git
   git sparse-checkout init
   git sparse-checkout set "Task3/Hands Sign"

   # Pull the content
   git pull origin main
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data**:
   download the data from [HERE](https://universe.roboflow.com/vishaal-w63ex/signs-n3pju-zgvif).

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Select Input Method**:
   - **Webcam**: Start the camera for real-time gesture recognition.
   - **Upload Image**: Choose an image from your device to classify a sign.

3. **Adjust Confidence Level**:
   Adjust the confidence threshold in the sidebar to set the minimum confidence level for predictions.

## Model Training

The model was trained on MediaPipe-generated hand landmarks, using a neural network with dense layers, dropout, and batch normalization. The architecture is optimized for efficient real-time prediction.

### To Train the Model

1. Place images and labels in the following directory structure:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. Open `train_model.py` to preprocess data, train the model, and evaluate performance. Save the model after training:
   ```python
   model.save('sign_language_model.h5')
   ```

## Evaluation

The trained model achieved an accuracy of approximately **96.5%** on the validation set. The evaluation includes a confusion matrix and classification report to analyze model performance.

## Visualization

The training history for accuracy and loss is plotted to visualize model improvement over epochs. A confusion matrix provides insights into the model's performance across different classes.

---

Let me know if you want to add any specific details or images for visualization!
