# Plant Disease Classification 🌿

A deep learning-based plant disease classification system that can identify various plant diseases from leaf images. The project includes both model training scripts and a user-friendly web interface built with Streamlit.

## Overview

This project uses deep learning to classify plant diseases from leaf images, supporting 38 different classes including both healthy and diseased states across various plant species. The system can detect diseases in plants such as:
- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

## App Deployment

We used hugging face 🤗 to deploy the streamlit app and make use of the ZeroGPU feature to use the model faster.
- [App on Hugging Face](https://huggingface.co/spaces/ziadmostafa/Plants-Diseases-Classification)

## Model Performance

We evaluated several deep learning architectures on the plant disease dataset:

| Model | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|-------|---------------|---------------|-----------------|----------------|--------------|-------------|
| Simple CNN | 0.999 | 0.974 | 0.999 | 0.975 | 0.999 | 0.973 |
| ResNet50 | 0.994 | 0.980 | 0.994 | 0.981 | 0.994 | 0.979 |
| EfficientNetB0 | 1.000 | 0.997 | 1.000 | 0.997 | 1.000 | 0.997 |
| MobileNetV2 | 0.985 | 0.435 | 0.987 | 0.450 | 0.984 | 0.428 |
| DenseNet121 | 0.999 | 0.997 | 0.999 | 0.997 | 0.999 | 0.996 |

Based on the results, EfficientNetB0 and DenseNet121 show the best performance with 99.7% test accuracy.
