# 🚗 Vehicle Speed Prediction with Machine Learning

This project aims to estimate a vehicle's speed (km/h) using engine RPM and throttle position (%) data. It is built using real-world vehicle telematics data and machine learning techniques.

---
## 🗂 Dataset Source

This project uses real-world vehicle telemetry data from the following Kaggle dataset:

🔗 [Levin Vehicle Telematics Dataset – by yunlevin](https://www.kaggle.com/datasets/yunlevin/levin-vehicle-telematics)

## 🎯 Objective

- **Input:** Engine RPM + Throttle Position (%)
- **Output:** Predicted Vehicle Speed (km/h)
- **Goal:** Develop a reliable ML model with low prediction error for near real-time vehicle speed estimation.

---

## 📊 Techniques Used

- 🔹 **Data Preprocessing:**  
  - Cleaning missing and abnormal values  
  - Converting fields to numeric (e.g. `rpm`, `tPos`, `gps_speed`)

- 🔹 **Anomaly Detection:**  
  - `Isolation Forest` was used to detect and remove outliers  
  - Helps the model learn cleaner and more stable patterns

- 🔹 **Device Segmentation:**  
  - Grouped by `deviceID`, and top-performing devices (3, 9, 12) were selected  
  - Model was retrained using only high-quality data

- 🔹 **Modeling:**  
  - `RandomForestRegressor`  
  - Optimized using `GridSearchCV`  
  - Best Parameters:
    - `n_estimators`: 200  
    - `max_depth`: 20  
    - `min_samples_split`: 2  
    - `min_samples_leaf`: 1  

---

## 📈 Model Performance
📦 **Download Model:**  
You can download the machine learning model from the link below:

🔗 [Download Vehicle Speed Prediction Model (.pkl)](https://drive.google.com/uc?id=1YKG7j9f57xLVvK9Q5UpKiIHZ0dj5MqxM)

> Note: Hosted on Google Drive due to GitHub file size limitations.


- 🎯 **R² Score:** 0.627  
- 📉 **Mean Absolute Error (MAE):** 9.90 km/h  
- 🧪 Prediction Range: **500 – 4000 RPM**

---

## ⚠️ Limitations & Considerations

Although the model performs well in the defined range, the following issues are noted:

- ❗ **RPM > 2500 values** were excluded due to data sparsity and unreliability
- ⚠️ Even within the 500–2500 RPM range, some inconsistencies exist due to:
  - Sensor irregularities
  - Constant throttle values for some devices
  - Missing features like slope, load, or environmental factors

- 🔧 Recommendations:
  - Engineer new features (e.g. `rpm/tPos` ratio, acceleration)
  - Test other models (e.g. XGBoost, LightGBM)
  - Add time-series awareness with RNNs or LSTMs

---

## 🖥️ Streamlit Interface

The model was deployed using Streamlit for an interactive user experience.

Users can:
- Input RPM and throttle values
- Instantly receive speed predictions

---



## 👤 Author

**Mert Yenigün**  
[GitHub Profile](https://github.com/mertyenigun)
