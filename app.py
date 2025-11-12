import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
import time

# -----------------------------------------------------
# 🔹 File Paths (Update These)
# -----------------------------------------------------
MODEL_PATH = r"C:\Users\rosha\Downloads\project 5\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\models\best_model.h5"
CLASS_PATH = r"C:\Users\rosha\Downloads\project 5\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\models\class_indices.json"

# -----------------------------------------------------
# 🔹 Load Model and Class Labels
# -----------------------------------------------------
st.sidebar.title("⚙️ App Settings")

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH):
    model = load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        class_indices = json.load(f)
    inv_map = {v: k for k, v in class_indices.items()}
else:
    st.error("⚠️ Model or class index file not found. Please check file paths below!")
    st.write("🔍 Current Working Directory:", os.getcwd())
    st.stop()

# -----------------------------------------------------
# 🔹 Streamlit App UI
# -----------------------------------------------------
st.title("🐟 Fish Classification App")
st.write("Upload an image to predict the fish species using a trained deep learning model.")

# Sidebar info
st.sidebar.header("📘 Model Information")
st.sidebar.write("**Model:** Custom CNN / Transfer Learning")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.write("**Input Size:** 224 × 224")
st.sidebar.write("**Classes:**", len(inv_map))
st.sidebar.write("**Trained on:** Fish dataset")

# -----------------------------------------------------
# 🔹 Image Upload
# -----------------------------------------------------
uploaded = st.file_uploader("📸 Choose a fish image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='📷 Uploaded Image', use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction progress
    with st.spinner('🔍 Analyzing image... Please wait...'):
        time.sleep(1.5)
        preds = model.predict(img_array)[0]
        top_indices = preds.argsort()[-3:][::-1]  # top-3 predictions
        confidence_scores = [(inv_map[i], preds[i] * 100) for i in top_indices]
    
    st.success("✅ Prediction complete!")

    # -----------------------------------------------------
    # 🔹 Display Top Predictions
    # -----------------------------------------------------
    st.subheader("🎯 Prediction Results")
    st.write("Here are the top 3 most likely species:")

    for i, (label, conf) in enumerate(confidence_scores):
        st.markdown(f"**{i+1}. {label}** — {conf:.2f}% confidence")

    top_label, top_conf = confidence_scores[0]
    st.markdown("---")
    st.markdown(f"### 🐠 Final Prediction: **{top_label}**")
    st.markdown(f"### 🔹 Confidence: **{top_conf:.2f}%**")

    # -----------------------------------------------------
    # 🔹 Visualization
    # -----------------------------------------------------
    st.subheader("📊 Confidence Distribution")
    chart_data = pd.DataFrame(preds, index=list(inv_map.values()), columns=["Confidence"])
    st.bar_chart(chart_data)

    # -----------------------------------------------------
    # 🔹 Downloadable Results
    # -----------------------------------------------------
    result_df = pd.DataFrame(confidence_scores, columns=["Fish Species", "Confidence (%)"])
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name='fish_prediction_results.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Please upload a fish image to begin prediction.")
