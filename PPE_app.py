import streamlit as st
import numpy as np
from PIL import Image
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-compliant {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .status-noncompliant {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F44336;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:first-child {
        display: none;
    }
    .stBadge, [data-testid="stBadge"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_ppe_model():
    try:
        model = load_model("mobilenet_ppe_model.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_ppe_model()

# Ensure page state exists
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# ======================
# SIDEBAR NAVIGATION
# ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hard-hat.png", width=80)
    st.title("🪖 PPE Monitor")
    st.markdown("### Navigation")
   
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "🏠 Home"
    if st.button("🔍 Prediction", use_container_width=True):
        st.session_state.page = "🔍 Prediction"
    if st.button("📊 History", use_container_width=True):
        st.session_state.page = "📊 History"
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "ℹ️ About"
   
    page = st.session_state.page
    st.divider()

# ======================
# HOME PAGE
# ======================
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>PPE Compliance Monitor</h1>", unsafe_allow_html=True)
   
    st.success("👋 You are warmly welcome to this application!")
    col1, col2 = st.columns([2, 1])
   
    with col1:
        st.markdown("""
        ### Ensure Workplace Safety
       
        Our CNN-powered system detects whether workers are wearing safety helmets.
       
        **Key Features:**
        - Real-time helmet detection  
        - High accuracy MobileNetV2 model  
        - Adjustable decision threshold  
        - Instant compliance feedback  
        """)            
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Model Accuracy", "98.5%")
        st.metric("Model Type", "MobileNetV2")
        st.metric("Task", "Binary Classification")
   
    st.divider()
    st.markdown("### How it Works")
    cols = st.columns(3)
    with cols[0]:
        st.image("https://img.icons8.com/fluency/96/upload.png")
        st.subheader("1. Upload")
        st.write("Upload a clear image of the worker")
    with cols[1]:
        st.image("https://img.icons8.com/fluency/96/brain.png")
        st.subheader("2. Analyze")
        st.write("Process the image")
    with cols[2]:
        st.image("https://img.icons8.com/fluency/96/approval.png")
        st.subheader("3. Report")
        st.write("Get instant compliance status")

# ======================
# PREDICTION PAGE
# ======================
elif page == "🔍 Prediction":
    st.title("🔍 PPE Compliance Check")
    st.markdown("Upload an image to check helmet compliance.")
    col1, col2 = st.columns([1, 2])
   
    with col1:
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
       
        if uploaded_file and st.button("🔄 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                time.sleep(1)
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    display_img = img.copy()
                    img = img.resize((224, 224))
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                   
                    prediction = model.predict(img_array, verbose=0)[0][0]
                   
                    if prediction > threshold:
                        label = "Helmet"
                        confidence = prediction
                        compliant = True
                    else:
                        label = "No Helmet"
                        confidence = 1 - prediction
                        compliant = False
                   
                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file": uploaded_file.name,
                        "label": label,
                        "confidence": confidence,
                        "compliant": compliant
                    })
                   
                    st.session_state.result = {
                        "image": display_img,
                        "label": label,
                        "confidence": confidence,
                        "compliant": compliant
                    }
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
   
    with col2:
        if "result" in st.session_state:
            r = st.session_state.result
            st.image(r["image"])                    # ← Natural size (as requested)
            st.markdown("---")
            if r["compliant"]:
                st.markdown(f"""
                <div class="status-compliant">
                    <h2>🟢 COMPLIANT</h2>
                    <p>Helmet detected</p>
                    <p>Confidence: {r['confidence']*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("Worker is compliant")
            else:
                st.markdown(f"""
                <div class="status-noncompliant">
                    <h2>🔴 NON-COMPLIANT</h2>
                    <p>No helmet detected</p>
                    <p>Confidence: {r['confidence']*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.error("PPE violation detected")
            st.progress(float(r["confidence"]))
        else:
            st.info("Upload an image to begin.")

# ======================
# HISTORY & ABOUT
# ======================
elif page == "📊 History":
    st.title("📊 Prediction History")
    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        for h in reversed(st.session_state.history[-10:]):
            icon = "🟢" if h["compliant"] else "🔴"
            st.markdown(f"""
            {icon} **{h['time']}**  
            📁 File: `{h['file']}`  
            🏷️ Prediction: **{h['label']}**  
            🎯 Confidence: **{h['confidence']*100:.2f}%**
            """)
            st.divider()

elif page == "ℹ️ About":
    st.title("About")
    st.markdown("""
    This application uses a **MobileNetV2 CNN model** to detect helmet usage.
   
    **Details:**
    - Binary classification (Helmet / No Helmet)
    - Pretrained MobileNetV2 (Transfer Learning)
    - High accuracy (~98.5%)
   
    **Goal:** Improve workplace safety using AI.
    """)

st.markdown("---")
st.caption("PPE Compliance Monitor • CNN-Based System • Thrive Plus Project 2026")
