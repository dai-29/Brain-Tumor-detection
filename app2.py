import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torch import argmax
from torchvision.models import resnet50
import os

# Load the labels and model
LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
resnet_model = resnet50(pretrained=True)

# Modify the final layer to match the number of classes
n_inputs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, 2048),
    torch.nn.SELU(),
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(2048, 2048),
    torch.nn.SELU(),
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(2048, len(LABELS)),
    torch.nn.LogSigmoid()
)

# Load trained model weights

resnet_model = resnet_model.to(device)
resnet_model.eval()

# Define image preprocessing function
def preprocess_image(image: Image.Image):
    transform = Compose([Resize((512, 512)), ToTensor()])
    return transform(image).unsqueeze(0)

# Define prediction function
def get_prediction(image: Image.Image):
    tensor = preprocess_image(image)
    y_hat = resnet_model(tensor.to(device))
    class_id = argmax(y_hat.data, dim=1)
    return LABELS[int(class_id)]

# Streamlit app interface
st.title("Brain Tumor Detection")

st.write("Upload an MRI image, and the model will classify it as either tumor or non-tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Get and display the prediction
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction = get_prediction(image)
            st.write(f"Prediction: **{prediction}**")
