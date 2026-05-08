import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# ================= LOAD DATA =================
df = pd.read_csv("data.csv")

# ================= CLEAN DATA =================
df = df.drop(columns=[
    "Run_ID",
    "Notes"
])

# ================= CONVERT TEXT =================
df["Contamination (Y/N)"] = df["Contamination (Y/N)"].map({
    "Y": 1,
    "N": 0
})

# ================= REMOVE NULLS =================
df = df.dropna()

# ================= FEATURES =================
feature_columns = [
    "Avg_temp",
    "Avg_pH",
    "Min_DO %",
    "Inoculum size (mL)",
    "Max_RPM",
    "Total culture duration (days)",
    "Total feed of Lactose (mL)",
    "Contamination (Y/N)",
    "Total media volume (mL)"
]

X = df[feature_columns]

# ================= TARGET =================
y = df["Max FPU/ml"]

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# ================= TRAIN =================
model.fit(X_train, y_train)

# ================= PREDICT =================
predictions = model.predict(X_test)

# ================= SCORE =================
score = r2_score(y_test, predictions)

print(f"\n✅ R2 Score: {score:.2f}")

# ================= SAVE MODEL =================
pickle.dump(model, open("model.pkl", "wb"))

print("\n✅ model.pkl saved successfully")