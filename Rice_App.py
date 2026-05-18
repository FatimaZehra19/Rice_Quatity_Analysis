import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
import tempfile
import os
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
import torch.nn as nn

# Imports
from Baseline_CNN_Model import RiceCNN
from grad_cam import SimpleGradCAM

# Computer Vision utilities
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ==========================================
# CV PIPELINE FUNCTIONS
# ==========================================

def preprocess_image(image_path):
    """Convert image to binary mask using Otsu thresholding."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    processed = cv2.medianBlur(thresh, 3)
    cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(processed, [c], 0, 255, -1)

    return processed, image


def segment_grains(binary_image):
    """Watershed segmentation with per-component markers."""
    binary_u8 = (binary_image > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = binary_u8.astype(bool)

    distance = ndimage.distance_transform_edt(binary)
    _, cc_labels = cv2.connectedComponents(binary_u8)
    n_components = int(np.max(cc_labels))

    if n_components == 0:
        return cc_labels.astype(int), distance

    MIN_NOISE_AREA = 200
    component_areas = {
        lbl: int(np.sum(cc_labels == lbl))
        for lbl in range(1, n_components + 1)
        if int(np.sum(cc_labels == lbl)) >= MIN_NOISE_AREA
    }

    if not component_areas:
        return np.zeros_like(cc_labels, dtype=int), distance

    areas_sorted = sorted(component_areas.values())
    ref_idx = max(0, len(areas_sorted) // 4)
    single_grain_area = areas_sorted[ref_idx]
    MERGE_RATIO = 1.6

    final_markers = np.zeros(distance.shape, dtype=int)
    next_label = 1

    for lbl, area in component_areas.items():
        comp_mask = (cc_labels == lbl)
        comp_dist = distance * comp_mask
        comp_max = float(np.max(comp_dist))
        estimated_grains = area / single_grain_area

        if estimated_grains < MERGE_RATIO:
            r, c = np.unravel_index(np.argmax(comp_dist), comp_dist.shape)
            final_markers[r, c] = next_label
            next_label += 1
        else:
            min_dist_px = max(15, int(comp_max * 0.7))
            thresh_abs = comp_max * 0.55
            peaks = peak_local_max(
                comp_dist,
                min_distance=min_dist_px,
                threshold_abs=thresh_abs,
                labels=comp_mask,
                footprint=np.ones((3, 3)),
                exclude_border=False,
            )
            if len(peaks) == 0:
                r, c = np.unravel_index(np.argmax(comp_dist), comp_dist.shape)
                final_markers[r, c] = next_label
                next_label += 1
            else:
                for (r, c) in peaks:
                    final_markers[r, c] = next_label
                    next_label += 1

    labels = watershed(-distance, final_markers, mask=binary)
    return labels, distance


def extract_features(labels):
    """Extract grain measurements (area, length, aspect ratio)."""
    grain_features = []
    all_areas = []

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area > 0:
            all_areas.append(area)

    if len(all_areas) > 0:
        median_area = np.median(all_areas)
        MIN_AREA = max(200, 0.20 * median_area)
        MAX_AREA = 5 * median_area
    else:
        MIN_AREA = 200
        MAX_AREA = 10000

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        if len(cnt) >= 5:
            (x, y), (d1, d2), _ = cv2.fitEllipse(cnt)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            major_axis = max(w, h)
            minor_axis = min(w, h)
            x, y = int(x + w/2), int(y + h/2)

        aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
        grain_features.append({
            'label': int(label),
            'area': float(area),
            'length': float(major_axis),
            'width': float(minor_axis),
            'aspect_ratio': float(aspect_ratio),
            'centroid': (int(x), int(y))
        })

    return grain_features


# ==========================================
# UI CONFIG
# ==========================================
st.set_page_config(page_title="Rice Intelligence AI", layout="wide")

st.markdown("""
<style>
.stApp {background:#f8fafc;}
[data-testid="stSidebar"] {background:#ffffff;}
.card {
    background:#ffffff;
    padding:20px;
    border-radius:12px;
    border:1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - NAVIGATION & MODEL SELECT
# ==========================================
page = st.sidebar.radio("Navigation", ["Home", "Analysis"], key="nav_sidebar")
st.sidebar.divider()
st.sidebar.subheader("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select AI Architecture",
    ["Baseline CNN", "MobileNetV2 (Transfer)", "ResNet50 (Transfer)"]
)

# Model Information
with st.sidebar.expander("Model Details"):
    if model_choice == "Baseline CNN":
        st.write("**Type:** Custom 4-Layer CNN")
        st.write("**Status:** Baseline")
    elif model_choice == "MobileNetV2 (Transfer)":
        st.write("**Type:** Pretrained MobileNet_V2")
        st.write("**Focus:** Efficiency & Speed")
    else:
        st.write("**Type:** Pretrained ResNet50")
        st.write("**Focus:** High Accuracy")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model(selected_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    
    if selected_model == "Baseline CNN":
        model = RiceCNN()
        path = PROJECT_ROOT / "Experiments" / "rice_cnn_baseline_best.pth"
        model.load_state_dict(torch.load(path, map_location=device))
        target_layer = model.conv4
    
    elif selected_model == "MobileNetV2 (Transfer)":
        model = models.mobilenet_v2()
        # Reconstruct the custom head used in training
        last_channel = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        path = PROJECT_ROOT / "Experiments" / "rice_mobilenetv2_transfer_best.pth"
        model.load_state_dict(torch.load(path, map_location=device))
        target_layer = model.features[-1]
        
    elif selected_model == "ResNet50 (Transfer)":
        model = models.resnet50()
        # Reconstruct the custom head used in training
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        path = PROJECT_ROOT / "Experiments" / "rice_resnet50_transfer_best.pth"
        model.load_state_dict(torch.load(path, map_location=device))
        target_layer = model.layer4
        
    model = model.to(device).eval()
    grad_cam = SimpleGradCAM(model, target_layer)
    return model, grad_cam, device

model, grad_cam, device = load_model(model_choice)

CLASS_NAMES = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']

# ==========================================
# HOME PAGE
# ==========================================
if page == "Home":
    st.title("Rice Intelligence AI")
    st.subheader("Automated Rice Quality & Variety Analysis System")

    st.write("")

    # Hero Section
    st.markdown("""
    <div class="card">
    <h3>Smart Rice Inspection</h3>
    <p>This system uses computer vision and deep learning to analyze and classify rice varieties,
    with automatic grain counting and detailed measurements.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h4>Variety Classification</h4>
        <p>Identifies rice type using CNN model (5 varieties).</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h4>Grain Detection</h4>
        <p>Auto-counts grains using watershed segmentation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Explainable AI</h4>
        <p>Grad-CAM visualizes model decision making.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h4>Detailed Reports</h4>
        <p>Provides measurements and downloadable results.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("Go to Analysis tab to upload image and start.")

# ==========================================
# ANALYSIS PAGE
# ==========================================
elif page == "Analysis":

    st.title("Rice Analysis")

    uploaded = st.file_uploader("Upload rice image")

    if uploaded:
        # Write upload to a temp file and always clean it up afterwards
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            img_path = tmp.name
        finally:
            tmp.close()

        try:
            binary, original = preprocess_image(img_path)
            labels, distance = segment_grains(binary)
            features = extract_features(labels)

            # 1. Variety prediction
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item() * 100
            predicted_variety = CLASS_NAMES[pred]
            heatmap, _ = grad_cam.get_heatmap(tensor)

            total = len(features)
            avg_area = np.mean([g['area'] for g in features]) if features else 0
            avg_length = np.mean([g['length'] for g in features]) if features else 0

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Variety", predicted_variety)
            c2.metric("Confidence", f"{confidence:.2f}%")
            c3.metric("Grains Detected", total)
            c4.metric("Avg. Area (px²)", f"{avg_area:.0f}")

            st.divider()
            st.bar_chart({CLASS_NAMES[i]: float(probs[i].detach() * 100) for i in range(5)})
            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs(
                ["Variety Classification", "Grain Detection", "Measurements", "CV Pipeline"]
            )

            # TAB 1: Variety + Grad-CAM
            with tab1:
                col1, col2 = st.columns(2)
                col1.image(original, caption="Input Image")
                h = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
                h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(original, 0.6, h, 0.4, 0)
                col2.image(overlay, caption="Model Focus (Grad-CAM Heatmap)")
                st.success(f"✓ Model is **{confidence:.1f}%** confident this is **{predicted_variety}** rice.")

            # TAB 2: Grain detection overlay
            with tab2:
                vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                for g in features:
                    cv2.circle(vis, g['centroid'], 5, (0, 255, 0), -1)
                    cv2.putText(vis, str(g['label']),
                                (g['centroid'][0] + 8, g['centroid'][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                st.image(vis, caption="Detected Grains (Green Dots)")
                st.info(f"**Detection Summary:** Found **{total}** grains in this image.")

            # TAB 3: Measurements table + histograms
            with tab3:
                import pandas as pd
                df = pd.DataFrame(features)
                if not df.empty:
                    st.subheader("Grain Measurements")
                    st.dataframe(df[['label', 'area', 'length', 'width', 'aspect_ratio']], use_container_width=True)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Area Distribution**")
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        ax1.hist(df['area'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                        ax1.set_xlabel('Area (pixels²)')
                        ax1.set_ylabel('Frequency')
                        ax1.grid(alpha=0.3)
                        st.pyplot(fig1)
                        plt.close(fig1)
                    with col_b:
                        st.write("**Length Distribution**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        ax2.hist(df['length'], bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
                        ax2.set_xlabel('Length (pixels)')
                        ax2.set_ylabel('Frequency')
                        ax2.grid(alpha=0.3)
                        st.pyplot(fig2)
                        plt.close(fig2)
                    st.divider()
                    csv = df.to_csv(index=False).encode()
                    st.download_button("📥 Download Measurements CSV", csv, "rice_grain_measurements.csv")

            # TAB 4: CV pipeline visualisation
            with tab4:
                st.subheader("Computer Vision Pipeline")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(binary, caption="1️⃣ Binary Mask (Otsu)", use_container_width=True)
                with c2:
                    dist_viz = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
                    st.image(np.uint8(dist_viz), caption="2️⃣ Distance Transform", use_container_width=True)
                with c3:
                    if np.max(labels) > 0:
                        label_hue = np.uint8(179 * labels / np.max(labels))
                        blank_ch = 255 * np.ones_like(label_hue)
                        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
                        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
                        labeled_img[labels == 0] = 0
                        st.image(labeled_img, caption="3️⃣ Watershed Segmentation", use_container_width=True)
                st.divider()
                st.write("**Pipeline:** Gaussian Blur → Otsu Threshold → Morphological Clean-up → Distance Transform → Watershed Segmentation → Feature Extraction")
                st.info("This view shows how the AI 'sees' the grains. If the colors in Step 3 merge, the grains are too close for the current segmentation.")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

        finally:
            os.unlink(img_path)

    else:
        st.info("Upload image to begin analysis")

st.caption("MS Thesis Project - Rice Quality Analysis System")