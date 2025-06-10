import streamlit as st
import os
import cv2
import torch
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Ensure the necessary libraries are installed in the Streamlit environment
# In a production setup, you would handle dependencies in requirements.txt
# For demonstration, we'll assume they are available.

class SEMGrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Handle cases where directories might not exist in the deployed environment
        self.image_list = []
        if os.path.exists(image_dir):
            self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace(".png", "_mask.png"))

        # Use PIL for image loading in Streamlit apps
        try:
            image = Image.open(img_path).convert('L') # Convert to grayscale
            mask = Image.open(mask_path).convert('L') # Convert to grayscale
        except FileNotFoundError:
            st.warning(f"File not found: {img_path} or {mask_path}")
            return None, None
        except Exception as e:
            st.error(f"Error loading image or mask: {self.image_list[idx]}, Error: {e}")
            return None, None

        # Check if images or masks are empty (after loading)
        if image is None or mask is None:
             st.warning(f"Skipping empty or invalid image/mask: {self.image_list[idx]}")
             return None, None

        # Resize and convert to numpy arrays
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        # Add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.decoder2(torch.cat([self.upconv2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# Function to load the model
@st.cache_resource
def load_model():
    model = UNet()
    # Load the trained weights here.
    # You will need to save your model weights from the Colab training.
    # Example: model.load_state_dict(torch.load('model_weights.pth'))
    # For demonstration, we'll return the untrained model.
    # In a real app, you would load the saved weights.
    st.warning("Model weights are not loaded. Add `model.load_state_dict` to load your trained weights.")
    return model

# Streamlit App Layout
st.title("SEM Grain Segmentation")

st.write("This app demonstrates semantic segmentation of SEM images using a U-Net model.")

# Upload images
uploaded_file = st.file_uploader("Upload an SEM image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if opencv_image is not None:
        st.subheader("Original Image")
        st.image(opencv_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image
        resized_image = cv2.resize(opencv_image, (256, 256)) / 255.0
        input_tensor = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions

        # Load the model (cached)
        model = load_model()
        model.eval() # Set model to evaluation mode

        # Perform inference
        with torch.no_grad():
            # Assuming the model was trained on CPU or you handle device transfer
            # If your trained model was on GPU, move it to CPU for Streamlit deployment
            # model.to('cpu')
            prediction = model(input_tensor).squeeze().numpy()

        st.subheader("Segmentation Result")

        # Apply a threshold to the prediction to get a binary mask
        threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5)
        binary_prediction = (prediction > threshold).astype(np.uint8) * 255 # Convert to 0-255 for display

        # Display the prediction mask
        st.image(binary_prediction, caption=f"Segmentation Mask (Threshold: {threshold})", use_column_width=True, clamp=True)

        # Optional: Overlay the mask on the original image (resized)
        st.subheader("Overlay")
        original_resized_display = cv2.resize(opencv_image, (256, 256))
        # Create an RGBA image to overlay
        overlay_image = cv2.cvtColor(original_resized_display, cv2.COLOR_GRAY2RGBA)
        mask_color = np.zeros_like(overlay_image)
        # Assign a color (e.g., red) to the segmented area
        mask_color[:,:,0] = binary_prediction # Red channel
        mask_color[:,:,3] = binary_prediction * 0.5 # Alpha channel (transparency)

        # Combine original image and colored mask
        combined_image = cv2.addWeighted(overlay_image, 1, mask_color, 1, 0)

        st.image(combined_image, caption="Segmentation Overlay", use_column_width=True)


    else:
        st.error("Could not load the uploaded image.")

else:
    st.write("Please upload an image to get started.")
