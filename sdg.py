import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to load the model
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to extract image features using ResNet18
def extract_features(model, image):
    with torch.no_grad():
        features = model(image)
    return features.numpy()

# Function to calculate similarity
def calculate_similarity(features1, features2):
    return cosine_similarity(features1, features2)[0][0]

# Streamlit UI
st.title("Image Similarity Checker")
st.write("Upload two images to check their similarity using ResNet18.")

# File uploader
image1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
image2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

if image1_file and image2_file:
    image1 = Image.open(image1_file).convert("RGB")
    image2 = Image.open(image2_file).convert("RGB")

    st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)

    model = load_model()
    image1_tensor = preprocess_image(image1)
    image2_tensor = preprocess_image(image2)

    features1 = extract_features(model, image1_tensor)
    features2 = extract_features(model, image2_tensor)

    similarity = calculate_similarity(features1, features2)
    st.write(f"Similarity Score: {similarity:.2f}")

    if similarity > 0.8:
        st.success("The images are highly similar.")
    elif similarity > 0.5:
        st.warning("The images are moderately similar.")
    else:
        st.error("The images are not similar.")
