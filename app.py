import streamlit as st
import os
import cv2
import torch
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class SEMGrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = []
        if os.path.exists(image_dir):
            self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        filename, ext = os.path.splitext(img_name)
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, f"{filename}_mask{ext}")

        try:
            image = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            st.warning(f"File not found: {img_path} or {mask_path}")
            return None, None
        except Exception as e:
            st.error(f"Error loading image/mask: {img_name}, Error: {e}")
            return None, None

        if image is None or mask is None:
            st.warning(f"Skipping invalid image/mask: {img_name}")
            return None, None

        # Resize and convert
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

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


# Load model
@st.cache_resource
def load_model():
    model = UNet()
    st.warning("Model weights are not loaded. Use `model.load_state_dict(...)` to load trained weights.")
    return model


# Streamlit UI
st.title("SEM Grain Segmentation")
st.write("Segment grains in SEM images using a trained U-Net model.")

uploaded_file = st.file_uploader("Upload an SEM image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if opencv_image is not None:
        st.subheader("Original Image")
        st.image(opencv_image, caption="Uploaded Image", use_container_width=True)

        resized_image = cv2.resize(opencv_image, (256, 256)) / 255.0
        input_tensor = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        model = load_model().cpu()
        model.eval()

        with torch.no_grad():
            prediction = model(input_tensor).squeeze().numpy()

        st.subheader("Segmentation Result")
        threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5)
        binary_prediction = (prediction > threshold).astype(np.uint8) * 255

        st.image(binary_prediction, caption=f"Mask (Threshold: {threshold})", use_container_width=True)

        # Overlay
        original_resized_display = cv2.resize(opencv_image, (256, 256))
        overlay_image = cv2.cvtColor(original_resized_display, cv2.COLOR_GRAY2RGBA)
        mask_color = np.zeros_like(overlay_image)
        mask_color[:, :, 0] = binary_prediction  # Red channel
        mask_color[:, :, 3] = binary_prediction * 0.5  # Alpha

        combined_image = cv2.addWeighted(overlay_image, 1, mask_color, 1, 0)
        st.image(combined_image, caption="Overlay", use_container_width=True)

    else:
        st.error("Could not load the uploaded image.")

else:
    st.write("Please upload an image to begin.")
