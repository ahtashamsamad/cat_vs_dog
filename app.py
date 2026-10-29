import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

st.set_page_config(page_title="3D Pet Classifier | AI Studio", page_icon="🐶", layout="wide")

# Warm Beige & 3D Tactile Claymorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; color: #2C221E; }
    .stApp { background-color: #F5EFE6 !important; background-image: radial-gradient(at 10% 10%, #FAF6F0 0%, #F5EFE6 60%, #E8DFD1 100%) !important; color: #2C221E !important; }
    section[data-testid="stSidebar"] { background-color: #EDE4D7 !important; border-right: 2px solid #DFD3C3 !important; }
    section[data-testid="stSidebar"] * { color: #2C221E !important; }
    .beige-hero { background: #FFFDF9; border: 2px solid #E6DEC8; border-radius: 28px; padding: 28px 38px; margin-bottom: 28px; box-shadow: 0 14px 28px rgba(139, 115, 85, 0.10), 0 6px 0px #DFD3C3; display: flex; justify-content: space-between; align-items: center; }
    .badge-3d { background: #F4E3D3; color: #C86D51; padding: 6px 16px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; display: inline-block; margin-bottom: 10px; box-shadow: inset 0 2px 4px rgba(255,255,255,0.8), 0 2px 0 #DFC8B4; }
    .hero-title { font-size: 34px; font-weight: 800; color: #2C221E; margin: 0; }
    .hero-sub { color: #7A6B5D; font-size: 15px; margin-top: 6px; font-weight: 500; }
    .card-3d { background: #FFFDF9; border: 2px solid #E6DEC8; border-radius: 26px; padding: 26px; box-shadow: 0 12px 24px rgba(139, 115, 85, 0.08), 0 6px 0px #DFD3C3; margin-bottom: 24px; }
    div[data-testid="stFileUploader"] section { background-color: #FAF6F0 !important; border: 2px dashed #C8B6A6 !important; border-radius: 20px !important; padding: 22px !important; color: #5C4E43 !important; }
    div.stButton > button[kind="primary"] { background: linear-gradient(180deg, #E07A5F 0%, #C85A32 100%) !important; color: #FFFFFF !important; border: none !important; border-radius: 18px !important; padding: 14px 28px !important; font-size: 17px !important; font-weight: 700 !important; box-shadow: 0 6px 0px #9A3F1E, 0 10px 20px rgba(200, 90, 50, 0.25) !important; }
    .result-3d-box { background: linear-gradient(145deg, #FAF6F0, #F2E9DC); border: 2px solid #DFD3C3; border-radius: 24px; padding: 28px 20px; text-align: center; box-shadow: inset 0 3px 6px rgba(255,255,255,0.9), 0 6px 0px #D5C5B2; }
    .result-tag { font-size: 13px; font-weight: 700; color: #8C7A6B; text-transform: uppercase; letter-spacing: 1.5px; }
    .result-pet-name { font-size: 44px; font-weight: 800; color: #2C221E; margin: 10px 0; }
    .conf-pill-3d { background: #E3F2E6; border: 2px solid #A3D9B1; color: #256D3B; padding: 8px 22px; border-radius: 999px; font-size: 16px; font-weight: 800; display: inline-block; box-shadow: 0 3px 0px #A3D9B1; }
    .stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 20px; }
    .stat-pill { background: #FFFDF9; border: 1.5px solid #E6DEC8; border-radius: 16px; padding: 12px; box-shadow: 0 4px 0px #E6DEC8; }
    .stat-val { font-size: 18px; font-weight: 800; color: #C86D51; }
    .stat-lbl { font-size: 12px; font-weight: 600; color: #8C7A6B; }
    .warm-bar-bg { background: #E8DFD1; border-radius: 14px; height: 16px; width: 100%; margin: 8px 0 18px 0; overflow: hidden; }
    .warm-bar-dog { background: linear-gradient(90deg, #E07A5F, #F2A65A); height: 100%; border-radius: 14px; }
    .warm-bar-cat { background: linear-gradient(90deg, #81B29A, #529677); height: 100%; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cnn_model(path):
    return tf.keras.models.load_model(path)

with st.sidebar:
    st.markdown("## 🐾 3D Pet Studio")
    active_model_path = "cats_vs_dogs_cnn.keras"
    
    model = None
    if os.path.exists(active_model_path):
        model = load_cnn_model(active_model_path)
        st.success("✅ CNN Model Loaded!")
    else:
        st.error("⚠️ Model file nahi mili!")

    st.divider()
    st.markdown("### 🔧 Calibration Controls")
    # Agar model ki probability thori shift ho rahi ho toh threshold adjust kar sakein
    threshold = st.slider("Decision Threshold (Cat vs Dog)", 0.10, 0.90, 0.50, 0.05)
    invert_labels = st.checkbox("🔄 Invert Labels (If 0=Dog, 1=Cat)", value=False)

st.markdown("""
<div class="beige-hero">
    <div>
        <span class="badge-3d">✨ 3D AI Pet Classifier</span>
        <h1 class="hero-title">Cats vs Dogs Vision Studio</h1>
        <p class="hero-sub">Exact TensorFlow-matched preprocessing for accurate real-time CNN inference.</p>
    </div>
    <div style="font-size: 50px; background: #FAF6F0; padding: 12px 22px; border-radius: 22px; border: 2px solid #E6DEC8; box-shadow: 0 5px 0 #DFD3C3;">
        🐶🐱
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.05, 1], gap="large")

with col1:
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    st.markdown("### 📸 1. Upload Pet Photo")
    uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        predict_btn = st.button("🚀 Classify Pet Now", type="primary", use_container_width=True)
    else:
        predict_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    st.markdown("### 🎯 2. AI Classification Result")

    if uploaded_file is not None and model is not None:
        t0 = time.time()
        
        # EXACT TENSORFLOW TRAINING PREPROCESSING MATCH:
        img_array = np.array(image, dtype=np.float32)
        img_tensor = tf.image.resize(img_array, [256, 256]) # Bilinear TF Resize (Same as image_dataset_from_directory)
        img_tensor = img_tensor / 255.0
        img_batch = tf.expand_dims(img_tensor, axis=0)
        
        # Inference mode (training=False keeps BatchNormalization stable)
        raw_prob = float(model(img_batch, training=False).numpy()[0][0])
        if invert_labels:
            raw_prob = 1.0 - raw_prob
            
        latency_ms = int((time.time() - t0) * 1000)
        
        dog_pct = raw_prob * 100
        cat_pct = (1.0 - raw_prob) * 100
        
        if raw_prob >= threshold:
            pet_title = "It's a DOG! 🐶"
            conf_val = dog_pct
        else:
            pet_title = "It's a CAT! 🐱"
            conf_val = cat_pct

        st.markdown(f"""
        <div class="result-3d-box">
            <div class="result-tag">CNN Model Prediction</div>
            <div class="result-pet-name">{pet_title}</div>
            <div class="conf-pill-3d">✨ {conf_val:.2f}% Confidence</div>
            
            <div class="stats-row">
                <div class="stat-pill"><div class="stat-val">{latency_ms} ms</div><div class="stat-lbl">Speed</div></div>
                <div class="stat-pill"><div class="stat-val">256×256</div><div class="stat-lbl">TF Tensor</div></div>
                <div class="stat-pill"><div class="stat-val">{raw_prob:.4f}</div><div class="stat-lbl">Raw Sigmoid</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 📊 Probability Breakdown")
        st.markdown(f"""
        <div>
            <div style="display:flex; justify-content:space-between; font-weight:700; font-size:15px;">
                <span>🐶 Dog Probability</span><span style="color:#C85A32;">{dog_pct:.1f}%</span>
            </div>
            <div class="warm-bar-bg"><div class="warm-bar-dog" style="width: {dog_pct}%;"></div></div>
            <div style="display:flex; justify-content:space-between; font-weight:700; font-size:15px;">
                <span>🐱 Cat Probability</span><span style="color:#529677;">{cat_pct:.1f}%</span>
            </div>
            <div class="warm-bar-bg"><div class="warm-bar-cat" style="width: {cat_pct}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Left side par photo upload karein.")
    st.markdown('</div>', unsafe_allow_html=True)
