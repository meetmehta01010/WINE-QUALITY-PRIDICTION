import numpy
import pandas
import sklearn
import pyarrow
import streamlit

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyArrow:", pyarrow.__version__)
print("Streamlit:", streamlit.__version__)

import os
os.environ["STREAMLIT_DATAFRAME_SERIALIZATION"] = "legacy"  # Avoid Arrow/NumPy issues

import streamlit as st
from PIL import Image
import numpy as np
#import easyocr
import joblib
import re

# Load models
wine_quality_model = joblib.load("wine_quality_alcohol_model.pkl")
wine_vs_beer_model = joblib.load("wine_vs_beer_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#reader = easyocr.Reader(['en'])

st.title("🍷 Wine Label Analyzer")

input_method = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Wine Bottle Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # OCR extraction
    img_array = np.array(image)
    result = reader.readtext(img_array, detail=0)
    extracted_text = " ".join(result)
    st.subheader("Extracted Text from Label:")
    st.write(extracted_text)

    # Extract alcohol percentage from text (default 0.0 if not found)
    alcohol = 0.0
    match = re.search(r"(\d+\.?\d*)\s*%", extracted_text)
    if match:
        alcohol = float(match.group(1))
    st.write(f"Detected Alcohol Content: {alcohol}%")

    # Prepare feature for quality prediction
    features = np.array([[alcohol]])

    # Predict Wine vs Beer
    X_text = vectorizer.transform([extracted_text])
    type_pred = wine_vs_beer_model.predict(X_text)
    type_prob = wine_vs_beer_model.predict_proba(X_text)

    st.subheader("Prediction Results:")

    if type_pred[0] == 1:
        st.success(f"🍷 This is Real Wine (Confidence: {max(type_prob[0]) * 100:.2f}%)")

        # Predict wine quality based on alcohol only
        quality_pred = wine_quality_model.predict(features)
        quality_prob = wine_quality_model.predict_proba(features)

        if quality_pred[0] == 1:
            st.success(f"✅ Good Quality Wine (Confidence: {max(quality_prob[0]) * 100:.2f}%)")
        else:
            st.error(f"❌ Poor Quality Wine (Confidence: {max(quality_prob[0]) * 100:.2f}%)")
    else:
        st.warning(f"🍺 This is Beer (Confidence: {max(type_prob[0]) * 100:.2f}%)")