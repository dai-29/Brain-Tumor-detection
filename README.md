# Brain-Tumor-detection

Brain Tumor Detection (End-to-End)
Introduction
This project is a Flask web application for detecting brain tumors from MRI images using a deep learning model built with PyTorch. Users can upload MRI images through the app, and the model will classify them as either tumor or non-tumor. The goal of this project is to provide an intuitive interface for medical professionals to quickly identify potential brain tumors.

# Dataset:
The dataset contains MRI images, divided into two categories: tumor and non-tumor.
Preprocessing techniques are applied to the dataset to ensure optimal model performance.

# Project-Overview
This end-to-end project consists of:

Data Loading: Load MRI images for training, validation, and testing.
Data Preprocessing: Apply normalization, resizing, and augmentation techniques.
Model Building: Build a Convolutional Neural Network (CNN) using PyTorch to classify the MRI images.
Model Training: Train the model on GPU (if available) to detect brain tumors.
Flask Web Application: Develop a Flask app for user interaction, allowing image uploads for tumor detection.
Model Deployment: Deploy the trained model within the Flask app.
Prediction: Provide real-time predictions through the Flask web app.
