from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import io
from PIL import Image
import tensorflow as tf

app = FastAPI(title="AI Agriculture Assistant API")

# ==========================================================
# LOAD ALL MODELS (once, at startup)
# ==========================================================
print("Loading models, please wait...")

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

print("All models loaded successfully! API is ready.")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is alive and all models are loaded."}


@app.get("/")
def root():
    return {"message": "AI Agriculture Assistant API is running. Visit /test for an interactive dashboard."}


@app.get("/test", response_class=HTMLResponse)
def test_dashboard():
    return TEST_DASHBOARD_HTML


# ==========================================================
# 1. CROP RECOMMENDATION
# ==========================================================
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict/crop")
def predict_crop(data: CropInput):
    input_data = np.array([[data.N, data.P, data.K, data.temperature,
                             data.humidity, data.ph, data.rainfall]])
    prediction = crop_model.predict(input_data)
    crop_name = crop_label_encoder.inverse_transform(prediction)[0]
    return {"recommended_crop": crop_name}


# ==========================================================
# 2. FERTILIZER RECOMMENDATION
# ==========================================================
class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    potassium: float
    phosphorous: float

@app.post("/predict/fertilizer")
def predict_fertilizer(data: FertilizerInput):
    if data.soil_type not in fert_soil_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Invalid soil_type. Must be one of {list(fert_soil_encoder.classes_)}")
    if data.crop_type not in fert_crop_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Invalid crop_type. Must be one of {list(fert_crop_encoder.classes_)}")

    soil_encoded = fert_soil_encoder.transform([data.soil_type])[0]
    crop_encoded = fert_crop_encoder.transform([data.crop_type])[0]

    input_data = np.array([[data.temperature, data.humidity, data.moisture, soil_encoded,
                             crop_encoded, data.nitrogen, data.potassium, data.phosphorous]])
    prediction = fert_model.predict(input_data)
    fert_name = fert_name_encoder.inverse_transform(prediction)[0]
    return {"recommended_fertilizer": fert_name}


# ==========================================================
# 3. YIELD PREDICTION
# ==========================================================
class YieldInput(BaseModel):
    area: str
    item: str
    year: int
    rainfall: float
    pesticides: float
    avg_temp: float

@app.post("/predict/yield")
def predict_yield(data: YieldInput):
    if data.area not in yield_area_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Invalid area. Must be one of the trained countries.")
    if data.item not in yield_item_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Invalid item. Must be one of the trained crops.")

    area_encoded = yield_area_encoder.transform([data.area])[0]
    item_encoded = yield_item_encoder.transform([data.item])[0]

    input_data = np.array([[area_encoded, item_encoded, data.year, data.rainfall,
                             data.pesticides, data.avg_temp]])
    prediction = yield_model.predict(input_data)[0]
    return {"predicted_yield_hg_per_ha": round(float(prediction), 2)}


# ==========================================================
# 4. DISEASE PREDICTION (image upload)
# ==========================================================
@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = disease_model.predict(img_array)
    predicted_class = disease_class_names[int(np.argmax(prediction))]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return {"predicted_disease": predicted_class, "confidence_percent": confidence}


# ==========================================================
# SIMPLE BUILT-IN TEST DASHBOARD (no external CDN needed)
# ==========================================================
TEST_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI Agriculture Assistant - API Test Dashboard</title>
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #0e1117; color: #eee; margin: 0; padding: 30px; }
    h1 { color: #4CAF50; margin-bottom: 5px; }
    p.subtitle { color: #aaa; margin-top: 0; margin-bottom: 25px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .tab-btn { background: #1c2128; color: #ccc; border: none; padding: 12px 20px; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 600; font-size: 14px; }
    .tab-btn.active { background: #2E7D32; color: white; }
    .tab-content { display: none; background: #161b22; padding: 25px; border-radius: 0 8px 8px 8px; border: 1px solid #30363d; }
    .tab-content.active { display: block; }
    label { display: block; margin-top: 12px; margin-bottom: 4px; font-size: 13px; color: #ccc; }
    input, select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #444; background: #0e1117; color: #eee; box-sizing: border-box; }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
    button.submit-btn { margin-top: 20px; background: #2E7D32; color: white; border: none; padding: 12px 25px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 15px; width: 100%; }
    button.submit-btn:hover { background: #1B5E20; }
    .result-box { margin-top: 20px; padding: 15px; border-radius: 8px; background: #1b3a1e; border-left: 5px solid #4CAF50; font-weight: 600; display: none; }
    .result-box.error { background: #3a1b1b; border-left-color: #d32f2f; }
    .example-row { display: flex; gap: 10px; margin-bottom: 15px; }
    .example-btn { background: #21262d; color: #9fcf9f; border: 1px solid #2E7D32; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
</style>
</head>
<body>

<h1>🌱 AI Agriculture Assistant</h1>
<p class="subtitle">API Test Dashboard — try each endpoint directly, no external tools needed.</p>

<div class="tabs">
    <button class="tab-btn active" onclick="showTab('crop')">🌾 Crop</button>
    <button class="tab-btn" onclick="showTab('fert')">🧪 Fertilizer</button>
    <button class="tab-btn" onclick="showTab('yield')">📈 Yield</button>
    <button class="tab-btn" onclick="showTab('disease')">🍃 Disease</button>
</div>

<!-- CROP TAB -->
<div id="crop" class="tab-content active">
    <div class="example-row">
        <button class="example-btn" onclick="fillCrop(90,42,43,20.9,82,6.5,202.9)">💡 Example 1: Rice</button>
        <button class="example-btn" onclick="fillCrop(71,54,16,22.6,63.7,5.7,87.8)">💡 Example 2: Maize</button>
    </div>
    <div class="grid">
        <div><label>Nitrogen (N)</label><input id="c_N" type="number" value="90"></div>
        <div><label>Phosphorus (P)</label><input id="c_P" type="number" value="42"></div>
        <div><label>Potassium (K)</label><input id="c_K" type="number" value="43"></div>
        <div><label>Temperature (°C)</label><input id="c_temp" type="number" value="21"></div>
        <div><label>Humidity (%)</label><input id="c_hum" type="number" value="82"></div>
        <div><label>Soil pH</label><input id="c_ph" type="number" step="0.1" value="6.5"></div>
        <div><label>Rainfall (mm)</label><input id="c_rain" type="number" value="200"></div>
    </div>
    <button class="submit-btn" onclick="predictCrop()">🌾 Recommend Crop</button>
    <div id="c_result" class="result-box"></div>
</div>

<!-- FERTILIZER TAB -->
<div id="fert" class="tab-content">
    <div class="example-row">
        <button class="example-btn" onclick="fillFert(26,52,38,'Sandy','Maize',37,0,0)">💡 Example 1: Sandy + Maize</button>
        <button class="example-btn" onclick="fillFert(26,52,35,'Sandy','Barley',12,10,13)">💡 Example 2: Sandy + Barley</button>
    </div>
    <div class="grid">
        <div><label>Temperature (°C)</label><input id="f_temp" type="number" value="26"></div>
        <div><label>Humidity (%)</label><input id="f_hum" type="number" value="52"></div>
        <div><label>Moisture (%)</label><input id="f_moist" type="number" value="38"></div>
        <div><label>Soil Type</label><input id="f_soil" type="text" value="Sandy"></div>
        <div><label>Crop Type</label><input id="f_crop" type="text" value="Maize"></div>
        <div><label>Nitrogen</label><input id="f_n" type="number" value="37"></div>
        <div><label>Potassium</label><input id="f_k" type="number" value="0"></div>
        <div><label>Phosphorous</label><input id="f_p" type="number" value="0"></div>
    </div>
    <button class="submit-btn" onclick="predictFert()">🧪 Recommend Fertilizer</button>
    <div id="f_result" class="result-box"></div>
</div>

<!-- YIELD TAB -->
<div id="yield" class="tab-content">
    <div class="example-row">
        <button class="example-btn" onclick="fillYield('Albania','Maize',1990,1485,121,16.37)">💡 Example 1: Albania Maize</button>
        <button class="example-btn" onclick="fillYield('India','Wheat',1990,1083,75000,25.58)">💡 Example 2: India Wheat</button>
    </div>
    <div class="grid">
        <div><label>Area / Country</label><input id="y_area" type="text" value="Albania"></div>
        <div><label>Crop Item</label><input id="y_item" type="text" value="Maize"></div>
        <div><label>Year</label><input id="y_year" type="number" value="1990"></div>
        <div><label>Rainfall (mm/year)</label><input id="y_rain" type="number" value="1485"></div>
        <div><label>Pesticides (tonnes)</label><input id="y_pest" type="number" value="121"></div>
        <div><label>Avg Temperature (°C)</label><input id="y_temp" type="number" step="0.1" value="16.37"></div>
    </div>
    <button class="submit-btn" onclick="predictYield()">📈 Predict Yield</button>
    <div id="y_result" class="result-box"></div>
</div>

<!-- DISEASE TAB -->
<div id="disease" class="tab-content">
    <label>Upload a leaf image</label>
    <input id="d_file" type="file" accept="image/*">
    <button class="submit-btn" onclick="predictDisease()">🍃 Predict Disease</button>
    <div id="d_result" class="result-box"></div>
</div>

<script>
function showTab(name) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(name).classList.add('active');
    event.target.classList.add('active');
}

function showResult(id, text, isError) {
    const box = document.getElementById(id);
    box.style.display = 'block';
    box.className = 'result-box' + (isError ? ' error' : '');
    box.innerText = text;
}

function fillCrop(N,P,K,t,h,ph,r) {
    document.getElementById('c_N').value=N; document.getElementById('c_P').value=P;
    document.getElementById('c_K').value=K; document.getElementById('c_temp').value=t;
    document.getElementById('c_hum').value=h; document.getElementById('c_ph').value=ph;
    document.getElementById('c_rain').value=r;
}
function fillFert(t,h,m,soil,crop,n,k,p) {
    document.getElementById('f_temp').value=t; document.getElementById('f_hum').value=h;
    document.getElementById('f_moist').value=m; document.getElementById('f_soil').value=soil;
    document.getElementById('f_crop').value=crop; document.getElementById('f_n').value=n;
    document.getElementById('f_k').value=k; document.getElementById('f_p').value=p;
}
function fillYield(area,item,year,rain,pest,temp) {
    document.getElementById('y_area').value=area; document.getElementById('y_item').value=item;
    document.getElementById('y_year').value=year; document.getElementById('y_rain').value=rain;
    document.getElementById('y_pest').value=pest; document.getElementById('y_temp').value=temp;
}

async function predictCrop() {
    const body = {
        N: parseFloat(document.getElementById('c_N').value),
        P: parseFloat(document.getElementById('c_P').value),
        K: parseFloat(document.getElementById('c_K').value),
        temperature: parseFloat(document.getElementById('c_temp').value),
        humidity: parseFloat(document.getElementById('c_hum').value),
        ph: parseFloat(document.getElementById('c_ph').value),
        rainfall: parseFloat(document.getElementById('c_rain').value)
    };
    try {
        const res = await fetch('/predict/crop', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        showResult('c_result', '✅ Recommended Crop: ' + data.recommended_crop, false);
    } catch (e) { showResult('c_result', '❌ ' + e.message, true); }
}

async function predictFert() {
    const body = {
        temperature: parseFloat(document.getElementById('f_temp').value),
        humidity: parseFloat(document.getElementById('f_hum').value),
        moisture: parseFloat(document.getElementById('f_moist').value),
        soil_type: document.getElementById('f_soil').value,
        crop_type: document.getElementById('f_crop').value,
        nitrogen: parseFloat(document.getElementById('f_n').value),
        potassium: parseFloat(document.getElementById('f_k').value),
        phosphorous: parseFloat(document.getElementById('f_p').value)
    };
    try {
        const res = await fetch('/predict/fertilizer', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        showResult('f_result', '✅ Recommended Fertilizer: ' + data.recommended_fertilizer, false);
    } catch (e) { showResult('f_result', '❌ ' + e.message, true); }
}

async function predictYield() {
    const body = {
        area: document.getElementById('y_area').value,
        item: document.getElementById('y_item').value,
        year: parseInt(document.getElementById('y_year').value),
        rainfall: parseFloat(document.getElementById('y_rain').value),
        pesticides: parseFloat(document.getElementById('y_pest').value),
        avg_temp: parseFloat(document.getElementById('y_temp').value)
    };
    try {
        const res = await fetch('/predict/yield', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        showResult('y_result', '✅ Estimated Yield: ' + data.predicted_yield_hg_per_ha + ' hg/ha', false);
    } catch (e) { showResult('y_result', '❌ ' + e.message, true); }
}

async function predictDisease() {
    const fileInput = document.getElementById('d_file');
    if (!fileInput.files.length) { showResult('d_result', '❌ Please choose an image first.', true); return; }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const res = await fetch('/predict/disease', {method:'POST', body: formData});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        showResult('d_result', '✅ Prediction: ' + data.predicted_disease + ' (Confidence: ' + data.confidence_percent + '%)', false);
    } catch (e) { showResult('d_result', '❌ ' + e.message, true); }
}
</script>

</body>
</html>
"""