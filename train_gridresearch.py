import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

df = df[df["deviceID"].isin([3, 9, 12])]

X = df[["rpm", "tPos"]]
y = df["gps_speed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n GridSearch ile Parametreler:", grid_search.best_params_)
print(f" R² Skoru: {r2:.3f}")
print(f" MAE: {mae:.2f} km/h")

joblib.dump(best_model, "hiz_tahmin_modeli_grid.pkl")

