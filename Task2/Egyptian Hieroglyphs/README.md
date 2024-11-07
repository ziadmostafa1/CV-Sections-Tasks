# Egyptian Hieroglyph Classifier

## Overview
The Egyptian Hieroglyph Classifier is a machine learning project designed to classify images of Egyptian hieroglyphs. The project leverages various deep learning models to achieve high accuracy in identifying different hieroglyphic symbols.

## Hugging Face App
You can try out the Egyptian Hieroglyph Classifier using the [Hugging Face App](https://huggingface.co/spaces/ziadmostafa/Egyptian-Hieroglyphs-Classification).

## Models Used
The project compares the performance of several deep learning models, including:
- Simple CNN
- ResNet50
- EfficientNetB0
- MobileNetV2
- DenseNet121

## Model Performance
The table below summarizes the performance of each model in terms of accuracy, precision, and recall on both training and test datasets:

| Model           | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall |
|-----------------|----------------|---------------|-----------------|----------------|--------------|-------------|
| Simple CNN      | 0.999265       | 0.958869      | 0.999633        | 0.973333       | 0.999265     | 0.938303    |
| ResNet50        | 0.993390       | 0.002571      | 0.993748        | 0.002571       | 0.992288     | 0.002571    |
| EfficientNetB0  | 0.991553       | 0.951157      | 0.991915        | 0.951157       | 0.991186     | 0.951157    |
| MobileNetV2     | 0.978333       | 0.002571      | 0.980819        | 0.002674       | 0.976497     | 0.002571    |
| DenseNet121     | 0.993022       | 0.989717      | 0.993015        | 0.989717       | 0.991921     | 0.989717    |
