# Brain-Tumor-Detection

## Introduction
This project is an end-to-end solution for detecting brain tumors from MRI images. Developed with a deep learning model using PyTorch, the system is deployed as a Flask web application. Users can upload MRI images through the interface, and the trained model will analyze and classify them, predicting if a tumor is present. The goal is to create a user-friendly tool for medical professionals to assist in quick, preliminary diagnostics.

## Dataset
The dataset includes MRI images labeled into two classes: **tumor** and **non-tumor**. Each image undergoes preprocessing for model optimization, including resizing, normalization, and data augmentation.

## Project Overview
The project consists of the following stages:

1. **Data Loading**: Organize MRI images into training, validation, and testing datasets.
2. **Data Preprocessing**: Apply image normalization, resizing, and augmentations (e.g., rotations, flips) to improve model performance and generalization.
3. **Model Architecture**: Build a Convolutional Neural Network (CNN) using PyTorch. The model leverages transfer learning with a ResNet-50 architecture, fine-tuned to classify MRI images effectively.
4. **Model Training**: Train the model on a GPU (if available) to leverage faster computation, optimizing accuracy in detecting brain tumors.
5. **Flask Web Application**: Develop an intuitive Flask-based web interface that allows users to upload MRI images and receive a classification result.
6. **Model Deployment**: Integrate the trained model within the Flask app for live, real-time predictions.
7. **Prediction and Visualization**: Provide real-time predictions through the Flask interface, displaying results and probabilities to assist in diagnosis.

## Additional Project Features
- **Error Handling**: Built-in error handling for unsupported file types or image sizes, ensuring a smooth user experience.
- **Model Explainability**: Potential integration of Grad-CAM or similar methods for visualizing model focus areas, enhancing interpretability.
- **Secure Access**: Configurations for secure access to the web app, protecting sensitive data.

This project aims to serve as a valuable resource in medical diagnostics by providing fast, accurate, and accessible tumor detection capabilities. The model's real-time predictions, paired with a user-friendly interface, make it a practical tool for healthcare professionals.
