import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import tensorflow as tf

st.set_page_config(page_title="AI Agriculture Assistant", page_icon="🌱", layout="wide")

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown("""
    <style>
        /* ===== App background ===== */
        .stApp {
            background: linear-gradient(180deg, #0d1f14 0%, #0e1117 100%);
        }

        /* ===== Header ===== */
        .main-title {
            font-size: 44px;
            font-weight: 800;
            background: linear-gradient(90deg, #66BB6A, #A5D6A7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 16px;
            color: #9aa79d;
            margin-bottom: 30px;
        }

        /* ===== Result box ===== */
        .result-box {
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #1b3a24, #16291d);
            border-left: 6px solid #66BB6A;
            font-size: 20px;
            font-weight: 600;
            color: #C8E6C9;
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }

        /* ===== Section headers ===== */
        h3, .stMarkdown h3 {
            color: #A5D6A7 !important;
        }

        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid #1e3324;
        }
        .stTabs [data-baseweb="tab"] {
            height: 52px;
            padding: 0px 22px;
            background-color: #14251a;
            color: #8fae95;
            border-radius: 10px 10px 0px 0px;
            font-weight: 600;
            font-size: 15px;
            border: 1px solid #1e3324;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #1c3524;
            color: #C8E6C9;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
            color: white !important;
            border: 1px solid #2E7D32 !important;
        }

        /* ===== Buttons ===== */
        .stButton > button {
            background: linear-gradient(135deg, #2E7D32, #1B5E20);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 10px 0px;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #388E3C, #2E7D32);
            box-shadow: 0 4px 14px rgba(46,125,50,0.4);
            color: white;
        }

        /* ===== Inputs ===== */
        .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #131c17 !important;
            color: #eaeaea !important;
            border-radius: 6px !important;
        }
        label, .stCaption, p {
            color: #cfd8d2 !important;
        }

        /* ===== Dataframe (History tab) ===== */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1e3324;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌱 AI Agriculture Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">An end-to-end intelligent system for crop, fertilizer, yield, and disease prediction.</p>', unsafe_allow_html=True)


# ==========================================================
# LOAD ALL MODELS (once, cached so app stays fast)
# ==========================================================
@st.cache_resource
def load_all_models():
    crop_model = joblib.load("models/crop_model.pkl")
    crop_label_encoder = joblib.load("models/crop_label_encoder.pkl")

    fert_model = joblib.load("models/fertilizer_model.pkl")
    fert_soil_encoder = joblib.load("models/fert_soil_encoder.pkl")
    fert_crop_encoder = joblib.load("models/fert_crop_encoder.pkl")
    fert_name_encoder = joblib.load("models/fert_name_encoder.pkl")

    yield_model = joblib.load("models/yield_model.pkl")
    yield_area_encoder = joblib.load("models/yield_area_encoder.pkl")
    yield_item_encoder = joblib.load("models/yield_item_encoder.pkl")

    disease_model = tf.keras.models.load_model("models/disease_model.keras")
    disease_class_names = joblib.load("models/disease_class_names.pkl")

    return (crop_model, crop_label_encoder, fert_model, fert_soil_encoder, fert_crop_encoder,
            fert_name_encoder, yield_model, yield_area_encoder, yield_item_encoder,
            disease_model, disease_class_names)

(crop_model, crop_label_encoder, fert_model, fert_soil_encoder, fert_crop_encoder,
 fert_name_encoder, yield_model, yield_area_encoder, yield_item_encoder,
 disease_model, disease_class_names) = load_all_models()


# ==========================================================
# HISTORY TRACKING (kept in memory for the current session)
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []

def log_prediction(module, inputs_dict, result):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Module": module,
        "Inputs": str(inputs_dict),
        "Result": result
    })


# ==========================================================
# TOP TAB NAVIGATION
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌾  Crop Recommendation",
    "🧪  Fertilizer Recommendation",
    "📈  Yield Prediction",
    "🍃  Disease Prediction",
    "📊  History"
])


# ==========================================================
# 1. CROP RECOMMENDATION
# ==========================================================
with tab1:
    st.subheader("Find the best crop for your field")
    st.caption("Enter your soil nutrients and local weather conditions below.")

    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("💡 Example 1: Rice conditions", use_container_width=True):
            st.session_state.update({"crop_N": 90.0, "crop_P": 42.0, "crop_K": 43.0,
                                      "crop_temp": 20.9, "crop_hum": 82.0, "crop_ph": 6.5, "crop_rain": 202.9})
            st.rerun()
    with ex2:
        if st.button("💡 Example 2: Maize conditions", use_container_width=True):
            st.session_state.update({"crop_N": 71.0, "crop_P": 54.0, "crop_K": 16.0,
                                      "crop_temp": 22.6, "crop_hum": 63.7, "crop_ph": 5.7, "crop_rain": 87.8})
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, value=st.session_state.get("crop_N", 90.0), key="crop_N")
        temperature = st.number_input("Temperature (°C)", value=st.session_state.get("crop_temp", 21.0), key="crop_temp")
        ph = st.number_input("Soil pH", value=st.session_state.get("crop_ph", 6.5), key="crop_ph")
    with col2:
        P = st.number_input("Phosphorus (P)", min_value=0.0, value=st.session_state.get("crop_P", 42.0), key="crop_P")
        humidity = st.number_input("Humidity (%)", value=st.session_state.get("crop_hum", 82.0), key="crop_hum")
    with col3:
        K = st.number_input("Potassium (K)", min_value=0.0, value=st.session_state.get("crop_K", 43.0), key="crop_K")
        rainfall = st.number_input("Rainfall (mm)", value=st.session_state.get("crop_rain", 200.0), key="crop_rain")

    st.write("")
    if st.button("🌾 Recommend Crop", use_container_width=True):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(input_data)
        crop_name = crop_label_encoder.inverse_transform(prediction)[0]
        st.markdown(f'<div class="result-box">✅ Recommended Crop: {crop_name.title()}</div>', unsafe_allow_html=True)

        inputs = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
        result_text = f"Recommended Crop: {crop_name.title()}"
        log_prediction("Crop Recommendation", inputs, result_text)

        report = f"AI Agriculture Assistant - Crop Recommendation Report\n\nInputs:\n" + \
                 "\n".join([f"{k}: {v}" for k, v in inputs.items()]) + f"\n\nResult:\n{result_text}"
        st.download_button("⬇️ Download Report", report, file_name="crop_recommendation_report.txt", use_container_width=True)


# ==========================================================
# 2. FERTILIZER RECOMMENDATION
# ==========================================================
with tab2:
    st.subheader("Get the right fertilizer for your crop")
    st.caption("Enter your field and soil conditions below.")

    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("💡 Example 1: Sandy soil + Maize", use_container_width=True):
            st.session_state.update({"fert_temp": 26.0, "fert_hum": 52.0, "fert_moist": 38.0,
                                      "fert_soil": "Sandy", "fert_crop": "Maize",
                                      "fert_n": 37.0, "fert_k": 0.0, "fert_p": 0.0})
            st.rerun()
    with ex2:
        if st.button("💡 Example 2: Sandy soil + Barley", use_container_width=True):
            st.session_state.update({"fert_temp": 26.0, "fert_hum": 52.0, "fert_moist": 35.0,
                                      "fert_soil": "Sandy", "fert_crop": "Barley",
                                      "fert_n": 12.0, "fert_k": 10.0, "fert_p": 13.0})
            st.rerun()

    soil_options = list(fert_soil_encoder.classes_)
    crop_options = list(fert_crop_encoder.classes_)

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature_f = st.number_input("Temperature (°C)", value=st.session_state.get("fert_temp", 26.0), key="fert_temp")
        soil_type = st.selectbox("Soil Type", soil_options,
                                  index=soil_options.index(st.session_state.get("fert_soil", soil_options[0])))
    with col2:
        humidity_f = st.number_input("Humidity (%)", value=st.session_state.get("fert_hum", 52.0), key="fert_hum")
        crop_type = st.selectbox("Crop Type", crop_options,
                                  index=crop_options.index(st.session_state.get("fert_crop", crop_options[0])))
    with col3:
        moisture = st.number_input("Moisture (%)", value=st.session_state.get("fert_moist", 38.0), key="fert_moist")

    col4, col5, col6 = st.columns(3)
    with col4:
        nitrogen = st.number_input("Nitrogen", min_value=0.0, value=st.session_state.get("fert_n", 37.0), key="fert_n")
    with col5:
        potassium = st.number_input("Potassium", min_value=0.0, value=st.session_state.get("fert_k", 0.0), key="fert_k")
    with col6:
        phosphorous = st.number_input("Phosphorous", min_value=0.0, value=st.session_state.get("fert_p", 0.0), key="fert_p")

    st.write("")
    if st.button("🧪 Recommend Fertilizer", use_container_width=True):
        soil_encoded = fert_soil_encoder.transform([soil_type])[0]
        crop_encoded = fert_crop_encoder.transform([crop_type])[0]

        input_data = np.array([[temperature_f, humidity_f, moisture, soil_encoded,
                                 crop_encoded, nitrogen, potassium, phosphorous]])
        prediction = fert_model.predict(input_data)
        fert_name = fert_name_encoder.inverse_transform(prediction)[0]
        st.markdown(f'<div class="result-box">✅ Recommended Fertilizer: {fert_name}</div>', unsafe_allow_html=True)

        inputs = {"temperature": temperature_f, "humidity": humidity_f, "moisture": moisture,
                  "soil_type": soil_type, "crop_type": crop_type, "nitrogen": nitrogen,
                  "potassium": potassium, "phosphorous": phosphorous}
        result_text = f"Recommended Fertilizer: {fert_name}"
        log_prediction("Fertilizer Recommendation", inputs, result_text)

        report = f"AI Agriculture Assistant - Fertilizer Recommendation Report\n\nInputs:\n" + \
                 "\n".join([f"{k}: {v}" for k, v in inputs.items()]) + f"\n\nResult:\n{result_text}"
        st.download_button("⬇️ Download Report", report, file_name="fertilizer_recommendation_report.txt", use_container_width=True)


# ==========================================================
# 3. YIELD PREDICTION
# ==========================================================
with tab3:
    st.subheader("Estimate your crop yield")
    st.caption("Enter regional and environmental details below.")

    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("💡 Example 1: Albania, Maize (1990)", use_container_width=True):
            st.session_state.update({"yield_area": "Albania", "yield_item": "Maize", "yield_year": 1990,
                                      "yield_rain": 1485.0, "yield_pest": 121.0, "yield_temp": 16.37})
            st.rerun()
    with ex2:
        if st.button("💡 Example 2: India, Wheat (1990)", use_container_width=True):
            st.session_state.update({"yield_area": "India", "yield_item": "Wheat", "yield_year": 1990,
                                      "yield_rain": 1083.0, "yield_pest": 75000.0, "yield_temp": 25.58})
            st.rerun()

    area_options = list(yield_area_encoder.classes_)
    item_options = list(yield_item_encoder.classes_)

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.selectbox("Area / Country", area_options,
                             index=area_options.index(st.session_state.get("yield_area", area_options[0])))
        rainfall_y = st.number_input("Average Rainfall (mm/year)", value=st.session_state.get("yield_rain", 1200.0), key="yield_rain")
    with col2:
        item = st.selectbox("Crop Item", item_options,
                             index=item_options.index(st.session_state.get("yield_item", item_options[0])))
        pesticides = st.number_input("Pesticides (tonnes)", value=st.session_state.get("yield_pest", 120.0), key="yield_pest")
    with col3:
        year = st.number_input("Year", min_value=1990, max_value=2030, value=st.session_state.get("yield_year", 2013), key="yield_year")
        avg_temp = st.number_input("Average Temperature (°C)", value=st.session_state.get("yield_temp", 20.0), key="yield_temp")

    st.write("")
    if st.button("📈 Predict Yield", use_container_width=True):
        area_encoded = yield_area_encoder.transform([area])[0]
        item_encoded = yield_item_encoder.transform([item])[0]

        input_data = np.array([[area_encoded, item_encoded, year, rainfall_y, pesticides, avg_temp]])
        prediction = yield_model.predict(input_data)[0]
        st.markdown(f'<div class="result-box">✅ Estimated Yield: {prediction:,.2f} hg/ha</div>', unsafe_allow_html=True)

        inputs = {"area": area, "item": item, "year": year, "rainfall": rainfall_y,
                  "pesticides": pesticides, "avg_temp": avg_temp}
        result_text = f"Estimated Yield: {prediction:,.2f} hg/ha"
        log_prediction("Yield Prediction", inputs, result_text)

        report = f"AI Agriculture Assistant - Yield Prediction Report\n\nInputs:\n" + \
                 "\n".join([f"{k}: {v}" for k, v in inputs.items()]) + f"\n\nResult:\n{result_text}"
        st.download_button("⬇️ Download Report", report, file_name="yield_prediction_report.txt", use_container_width=True)


# ==========================================================
# 4. DISEASE PREDICTION
# ==========================================================
with tab4:
    st.subheader("Detect disease from a leaf image")
    st.caption("Upload a clear photo of a single leaf.")

    st.info(
        "💡 **No image handy?** Try any photo from your own `data/disease_data/` folder — for example, "
        "an image from **Potato___Early_blight** or **Potato___healthy** — to see how the model responds "
        "to each class."
    )

    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🍃 Predict Disease", use_container_width=True):
                img_resized = image.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = disease_model.predict(img_array)
                predicted_class = disease_class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.markdown(
                    f'<div class="result-box">✅ Prediction: {predicted_class}<br>Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True
                )

                inputs = {"image_file": uploaded_file.name}
                result_text = f"Prediction: {predicted_class} (Confidence: {confidence:.1f}%)"
                log_prediction("Disease Prediction", inputs, result_text)

                report = f"AI Agriculture Assistant - Disease Prediction Report\n\nInputs:\n" + \
                         "\n".join([f"{k}: {v}" for k, v in inputs.items()]) + f"\n\nResult:\n{result_text}"
                st.download_button("⬇️ Download Report", report, file_name="disease_prediction_report.txt", use_container_width=True)


# ==========================================================
# 5. HISTORY
# ==========================================================
with tab5:
    st.subheader("Your prediction history")
    st.caption("Everything you've predicted this session, most recent first.")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Try any of the 4 tools above and they'll show up here.")
    else:
        history_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False)
        st.download_button("⬇️ Download Full History (CSV)", csv_data, file_name="prediction_history.csv", use_container_width=True)

        st.write("")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
