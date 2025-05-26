import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS styling
st.markdown("""
<style>
.stApp {
    background: #e6f2ff;
    color: #004080;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3 {
    color: #003366;
}
</style>
""", unsafe_allow_html=True)

# Load YOLO model
model = YOLO('weights/last.pt')

# Sidebar
st.sidebar.title("🌲 Forest Analysis Options")
option = st.sidebar.radio(
    "Choose an analysis:",
    ('Tree Coverage %', 'Deforestation Change Over Time', 'Tree Density Heatmap')
)

# --- Option 1: Tree Coverage %
if option == 'Tree Coverage %':
    st.title("🌲 Tree Coverage Percentage")
    st.write("Upload an image to estimate tree coverage percentage.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        height, width = img_np.shape[:2]

        st.image(img_np, caption="Uploaded Image", use_column_width=True)

        results = model.predict(img_np, conf=0.1)  # Lower confidence threshold to detect more trees
        masks = results[0].masks

        if masks and masks.xyn:
            binary_mask = np.zeros((height, width), dtype=np.uint8)

            for polygon in masks.xyn:
                pts = np.array(polygon * [width, height], dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], 1)

            tree_pixels = np.count_nonzero(binary_mask == 1)
            total_pixels = height * width
            coverage_percent = (tree_pixels / total_pixels) * 100
            coverage_percent = min(coverage_percent, 100.0)

            st.success(f"🌿 Estimated Tree Coverage: {coverage_percent:.2f}%")
            st.image(binary_mask * 255, caption="Tree Mask (Black & White)", use_column_width=True)

            overlay_img = img_np.copy()
            overlay_img[binary_mask == 1] = [0, 255, 0]
            st.image(overlay_img, caption="Detected Tree Regions", use_column_width=True)
        else:
            st.warning("No tree masks detected.")

# --- Option 2: Deforestation Comparison
elif option == 'Deforestation Change Over Time':
    st.title("🌲 Deforestation Change Over Time")
    st.write("Upload two images from different times to compare tree coverage.")

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Before Image", type=["jpg", "jpeg", "png"], key="before")
    with col2:
        after_file = st.file_uploader("After Image", type=["jpg", "jpeg", "png"], key="after")

    if before_file and after_file:
        img_before = Image.open(before_file).convert("RGB")
        img_after = Image.open(after_file).convert("RGB")

        np_before = np.array(img_before)
        np_after = np.array(img_after)

        h, w = np_before.shape[:2]
        st.subheader("📸 Uploaded Images")
        st.image([np_before, np_after], caption=["Before", "After"], width=300)

        res_before = model.predict(np_before, conf=0.1)[0]
        res_after = model.predict(np_after, conf=0.1)[0]

        def get_mask(res, width, height):
            if res.masks and res.masks.xyn:
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in res.masks.xyn:
                    pts = np.array(poly * [width, height], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                return mask
            else:
                return np.zeros((height, width), dtype=np.uint8)

        mask_before = get_mask(res_before, w, h)
        mask_after = get_mask(res_after, w, h)

        cov_before = (mask_before.sum() / (h * w)) * 100
        cov_after = (mask_after.sum() / (h * w)) * 100
        delta = cov_after - cov_before

        st.subheader("📊 Tree Coverage Comparison")
        col1, col2 = st.columns(2)
        col1.metric("Before", f"{cov_before:.2f}%")
        col2.metric("After", f"{cov_after:.2f}%", delta=f"{delta:.2f}%")

        st.subheader("🖼️ Tree Masks")
        st.image([mask_before * 255, mask_after * 255], caption=["Before Mask", "After Mask"], width=300)

        lost_trees = np.logical_and(mask_before == 1, mask_after == 0)
        overlay = np.array(img_after).copy()
        overlay[lost_trees] = [255, 0, 0]  # Red highlights tree loss

        st.subheader("🔥 Tree Loss Visualized (Red = Lost Trees)")
        st.image(overlay, use_column_width=True)
    else:
        st.info("Please upload both images to continue.")

# --- Option 3: Tree Density Heatmap
elif option == 'Tree Density Heatmap':
    st.title("🌲 Tree Density Heatmap")
    st.write("Upload a satellite image to visualize tree density across regions.")

    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"], key="heatmap")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        st.image(img_np, caption="Uploaded Image", use_column_width=True)

        result = model.predict(img_np, conf=0.1)[0]
        masks = result.masks

        if masks and masks.xyn:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            for polygon in masks.xyn:
                pts = np.array(polygon * [w, h], dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], 1)

            # Divide image into grid cells and count trees per cell
            grid_size = 20  # You can adjust this
            heatmap = np.zeros((h // grid_size, w // grid_size))

            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    y_end = min(y + grid_size, h)
                    x_end = min(x + grid_size, w)
                    patch = binary_mask[y:y_end, x:x_end]

                    y_idx = y // grid_size
                    x_idx = x // grid_size

                    if y_idx < heatmap.shape[0] and x_idx < heatmap.shape[1]:
                        heatmap[y_idx, x_idx] = np.count_nonzero(patch)

            st.subheader("📊 Tree Density Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap, cmap="YlGn", cbar=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No tree masks found to generate heatmap.")
