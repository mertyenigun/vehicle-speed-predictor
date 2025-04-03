import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


df1 = pd.read_csv("allcars.csv", low_memory=False)
df2 = pd.read_csv("v2.csv", low_memory=False)
common_cols = df1.columns.intersection(df2.columns)
df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)


for col in ["rpm", "tPos", "gps_speed", "deviceID"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["rpm", "tPos", "gps_speed", "deviceID"])


df = df[(df["rpm"] > 500) & (df["rpm"] < 4000)]
df = df[(df["gps_speed"] > 5) & (df["gps_speed"] < 120)]


selected_ids = [3, 9, 12]
df_filtered = df[df["deviceID"].isin(selected_ids)]


X = df_filtered[["rpm", "tPos"]]
y = df_filtered["gps_speed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 Yeni Model R² Skoru (Seçili Device'lar): {r2:.3f}")
print(f"📉 Yeni Model MAE: {mae:.2f} km/h")

joblib.dump(model, "hiz_tahmin_modeli_selected.pkl")
print("✅ Model başarıyla hiz_tahmin_modeli_selected.pkl olarak kaydedildi.")
