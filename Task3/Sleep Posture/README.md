# Sleep Posture Classification

This project is a machine learning application for classifying sleeping postures using images, powered by a deep learning model trained on MediaPipe pose landmarks and implemented with TensorFlow and Streamlit.

## Project Overview

The application classifies images into one of four sleeping postures:
1. Prone
2. Supine
3. To-Left
4. To-Right

Users can upload an image of a person sleeping, and the model will predict their posture with a confidence score.

## Table of Contents

- [Sleep Posture Classification](#sleep-posture-classification)
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
- **TensorFlow**: for model building and training.
- **MediaPipe**: for pose landmark extraction.
- **Streamlit**: for creating an interactive web interface.
- **OpenCV**: for image processing.

## Setup

1. **Clone this repository**:
   ```bash
   # Initialize git repo
    git init Sleep-Posture
    cd Sleep-Posture

    # Add remote and enable sparse checkout
    git remote add origin https://github.com/ziadmostafa1/CV-Sections-Tasks.git
    git sparse-checkout init
    git sparse-checkout set "Task3/Sleep Posture"

    # Pull the content
    git pull origin main
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data**:
   download the data from [HERE](https://universe.roboflow.com/sam-vcqdz/object-detection-ikxzz).

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image**:
   - Select an image to classify the sleeping posture.
   - Adjust the confidence threshold in the sidebar to control the minimum confidence level for predictions.

## Model Training

The model was trained using images and pose landmarks generated from MediaPipe. The training process is detailed in the provided Jupyter Notebook (`Sleeping posture notebook.ipynb`). The model structure includes several dense layers with dropout and batch normalization for stability.

### To Train the Model

1. Place images and labels in the required directory structure:
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

2. Open the Jupyter Notebook (`Sleeping posture notebook.ipynb`) and run the cells to preprocess data, train the model, and evaluate performance.

3. Save the model after training:
   ```python
   model.save('sleep_posture_model.h5')
   ```

## Evaluation

The trained model was evaluated on a test dataset, achieving an accuracy of **91.57%**. Evaluation includes a confusion matrix and a detailed classification report for assessing model performance.

## Visualization

The training history for both accuracy and loss is plotted to show model improvement over epochs. A confusion matrix provides insights into the model's performance across different classes.


![](output.png)
![](output2.png)

---
