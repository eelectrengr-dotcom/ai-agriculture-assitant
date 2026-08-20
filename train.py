import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("models", exist_ok=True)

# ==========================================================
# 1. CROP RECOMMENDATION MODEL  (XGBoost)
# ==========================================================
print("Training Crop Recommendation model...")

crop_df = pd.read_csv("data/Crop_recommendation.csv")

label_encoder_crop = LabelEncoder()
crop_df["label_encoded"] = label_encoder_crop.fit_transform(crop_df["label"])

X = crop_df.drop(["label", "label_encoded"], axis=1)
y = crop_df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crop_model = XGBClassifier(random_state=42, eval_metric="mlogloss")
crop_model.fit(X_train, y_train)

print(f"  Accuracy: {crop_model.score(X_test, y_test):.2%}")

joblib.dump(crop_model, "models/crop_model.pkl")
joblib.dump(label_encoder_crop, "models/crop_label_encoder.pkl")
print("  Saved to models/crop_model.pkl\n")


# ==========================================================
# 2. FERTILIZER RECOMMENDATION MODEL  (XGBoost)
# ==========================================================
print("Training Fertilizer Recommendation model...")

fert_df = pd.read_csv("data/Fertilizer_Prediction.csv")
fert_df.columns = fert_df.columns.str.strip()

soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_name_encoder = LabelEncoder()

fert_df["Soil Type"] = soil_encoder.fit_transform(fert_df["Soil Type"])
fert_df["Crop Type"] = crop_encoder.fit_transform(fert_df["Crop Type"])
fert_df["Fertilizer Name"] = fert_name_encoder.fit_transform(fert_df["Fertilizer Name"])

X = fert_df.drop("Fertilizer Name", axis=1)
y = fert_df["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fert_model = XGBClassifier(random_state=42, eval_metric="mlogloss")
fert_model.fit(X_train, y_train)

print(f"  Accuracy: {fert_model.score(X_test, y_test):.2%}")

joblib.dump(fert_model, "models/fertilizer_model.pkl")
joblib.dump(soil_encoder, "models/fert_soil_encoder.pkl")
joblib.dump(crop_encoder, "models/fert_crop_encoder.pkl")
joblib.dump(fert_name_encoder, "models/fert_name_encoder.pkl")
print("  Saved model + encoders to models/\n")


# ==========================================================
# 3. CROP YIELD PREDICTION MODEL  (LightGBM)
# ==========================================================
print("Training Crop Yield Prediction model...")

yield_df = pd.read_csv("data/yield_df.csv")

if "Unnamed: 0" in yield_df.columns:
    yield_df = yield_df.drop("Unnamed: 0", axis=1)

area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

yield_df["Area"] = area_encoder.fit_transform(yield_df["Area"])
yield_df["Item"] = item_encoder.fit_transform(yield_df["Item"])

X = yield_df.drop("hg/ha_yield", axis=1)
y = yield_df["hg/ha_yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

yield_model = LGBMRegressor(random_state=42)
yield_model.fit(X_train, y_train)

print(f"  R^2 score: {yield_model.score(X_test, y_test):.2%}")

joblib.dump(yield_model, "models/yield_model.pkl")
joblib.dump(area_encoder, "models/yield_area_encoder.pkl")
joblib.dump(item_encoder, "models/yield_item_encoder.pkl")
print("  Saved model + encoders to models/\n")


# ==========================================================
# 4. DISEASE PREDICTION MODEL  (Deep Learning - CNN)
# ==========================================================
print("Training Disease Prediction model...")

img_size = (128, 128)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    "data/disease_data",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "data/disease_data",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print(f"  Classes found: {class_names}")

normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

disease_model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

disease_model.compile(optimizer="adam",
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])

disease_model.fit(train_data, validation_data=val_data, epochs=10)

disease_model.save("models/disease_model.keras")
joblib.dump(class_names, "models/disease_class_names.pkl")
print("  Saved model + class names to models/\n")

print("All 4 models trained and saved successfully!")