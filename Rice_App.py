import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "Broken_Grains_Analysis"))
sys.path.append(str(PROJECT_ROOT / "src"))
import torch.nn as nn

# Imports
from Preprocessing import preprocess_image
from Segmentation import segment_grains
from Feature_Analysis import extract_features
from Classification import classify_grains
from Baseline_CNN_Model import RiceCNN

# ==========================================
# Grad-CAM
# ==========================================
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= (torch.max(heatmap) + 1e-10)
        return heatmap.detach().cpu().numpy(), idx

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
    <p>This system uses computer vision and deep learning to analyze rice grains,
    detect broken grains, and classify rice varieties.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h4>Variety Classification</h4>
        <p>Identifies rice type using CNN model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h4>Explainable AI</h4>
        <p>Grad-CAM shows model focus areas.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Broken Grain Detection</h4>
        <p>Detects damaged grains using image processing.</p>
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
        img_path = str(PROJECT_ROOT/"temp.jpg")
        with open(img_path,"wb") as f:
            f.write(uploaded.getbuffer())

        binary, original = preprocess_image(img_path)
        labels, distance = segment_grains(binary)
        features = extract_features(labels)
        
        # 1. First get Variety Prediction
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()*100
        predicted_variety = CLASS_NAMES[pred]

        # 2. Now Classify Quality (Full vs Broken) using Variety Info
        classified, (max_len, max_area) = classify_grains(features, variety_name=predicted_variety)

        heatmap, _ = grad_cam.get_heatmap(tensor)

        total = len(classified)
        full = sum(1 for g in classified if g['classification']=="Full")
        broken = total - full
        broken_perc = broken/total*100 if total else 0

        # Metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Variety", predicted_variety)
        c2.metric("Confidence", f"{confidence:.2f}%")
        c3.metric("Total Grains", total)
        c4.metric("Quality (Full)", f"{full} ({100-broken_perc:.1f}%)")

        st.divider()

        # Confidence Chart
        st.bar_chart({CLASS_NAMES[i]:float(probs[i].detach()*100) for i in range(5)})

        st.divider()

        tab1,tab2,tab3,tab4 = st.tabs(["Variety Insight","Quality Analysis","Measurements","Computer Perspective"])

        # TAB 1: Variety
        with tab1:
            col1,col2 = st.columns(2)
            col1.image(original, caption="Input Image")
            h = cv2.resize(heatmap,(original.shape[1],original.shape[0]))
            h = cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original,0.6,h,0.4,0)
            col2.image(overlay, caption="Deep Learning Focus (Grad-CAM)")
            st.info(f"Model is {confidence:.1f}% sure this is **{predicted_variety}** rice.")

        # TAB 2: Quality
        with tab2:
            vis = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            for g in classified:
                color = (0,255,0) if g['classification']=="Full" else (255,0,0)
                cv2.circle(vis, g['centroid'], 8, color, -1)
                cv2.putText(vis, str(g['label']), (g['centroid'][0]+10, g['centroid'][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            st.image(vis, caption="Green: Full Grain | Red: Broken/Damaged")
            st.write(f"**Analysis Summary:** Found {total} grains. {full} are full, {broken} are broken.")

        # TAB 3: Measurements
        with tab3:
            import pandas as pd
            df = pd.DataFrame(classified)
            if not df.empty:
                st.dataframe(df[['label', 'area', 'length', 'width', 'aspect_ratio', 'classification']])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Area Distribution**")
                    fig1, ax1 = plt.subplots()
                    df[df['classification']=='Full']['area'].hist(ax=ax1, alpha=0.5, label='Full', color='green')
                    df[df['classification']=='Broken']['area'].hist(ax=ax1, alpha=0.5, label='Broken', color='red')
                    ax1.legend()
                    st.pyplot(fig1)
                
                with col_b:
                    st.write("**Length Distribution**")
                    fig2, ax2 = plt.subplots()
                    df[df['classification']=='Full']['length'].hist(ax=ax2, alpha=0.5, label='Full', color='green')
                    df[df['classification']=='Broken']['length'].hist(ax=ax2, alpha=0.5, label='Broken', color='red')
                    ax2.legend()
                    st.pyplot(fig2)

                csv = df.to_csv(index=False).encode()
                st.download_button("Download Detailed Report", csv, "rice_quality_report.csv")

        # TAB 4: Computer Perspective (Thesis Enhancement)
        with tab4:
            st.subheader("Image Processing Pipeline")
            c1, c2, c3 = st.columns(3)
            
            c1.image(binary, caption="1. Binary Mask (Otsu)")
            
            # Distance Transform Viz
            dist_viz = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
            c2.image(np.uint8(dist_viz), caption="2. Distance Transform")
            
            # Label Viz
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[labels == 0] = 0
            c3.image(labeled_img, caption="3. Watershed Segmentation")
            
            st.info("This view shows how the AI 'sees' the grains. If the colors in Step 3 merge, it means the grains are too close for the current segmentation.")

    else:
        st.info("Upload image to begin analysis")

st.caption("MS Thesis Project - Rice Quality Analysis System")