# ============================================================
#   CROP RECOMMENDATION SYSTEM
#   Algorithms: Naive Bayes, Logistic Regression, Random Forest
#   Dataset   : Kaggle Crop Recommendation Dataset
#   XAI       : Feature Importance (Random Forest)
# ============================================================

# ── STEP 0: INSTALL REQUIRED LIBRARIES ───────────────────
import subprocess, sys

required = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
]

print("Checking and installing required libraries...")
for package in required:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "-q"]
    )
print("✅ All libraries ready.\n")


# ── STEP 1: IMPORT LIBRARIES ──────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

print("=" * 60)
print("   CROP RECOMMENDATION SYSTEM")
print("=" * 60)


# ── STEP 2: LOAD DATASET ──────────────────────────────────
# If using Google Colab upload the CSV and change path below
df = pd.read_csv("crop_recommendation.csv")

print("\n✅ Dataset Loaded Successfully")
print(f"   Total Records : {df.shape[0]}")
print(f"   Total Columns : {df.shape[1]}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nColumn Names  : {list(df.columns)}")
print(f"\nCrop Types    : {df['label'].nunique()} unique crops")
print(f"\nRecords per Crop:\n{df['label'].value_counts()}")


# ── STEP 3: CHECK DATA ────────────────────────────────────
print("\n── DATA QUALITY CHECK ──")
print(f"Missing Values :\n{df.isnull().sum()}")
print(f"\nData Types     :\n{df.dtypes}")
print(f"\nBasic Stats    :\n{df.describe()}")


# ── STEP 4: VISUALIZATIONS (EDA) ─────────────────────────

# 4A — Heatmap (Feature Correlation)
plt.figure(figsize=(10, 7))
numeric_df = df.drop('label', axis=1)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150)
plt.show()
print("✅ Heatmap saved as heatmap.png")

# 4B — Histogram (Data Distribution)
df.drop('label', axis=1).hist(figsize=(14, 10), bins=20,
                               color='steelblue', edgecolor='black')
plt.suptitle('Feature Distribution Histogram', fontsize=14,
             fontweight='bold')
plt.tight_layout()
plt.savefig('histogram.png', dpi=150)
plt.show()
print("✅ Histogram saved as histogram.png")

# 4C — Crop Count Bar Chart
plt.figure(figsize=(14, 6))
df['label'].value_counts().plot(kind='bar', color='teal', edgecolor='black')
plt.title('Number of Records per Crop', fontsize=14, fontweight='bold')
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('crop_count.png', dpi=150)
plt.show()
print("✅ Crop count chart saved as crop_count.png")


# ── STEP 5: PREPROCESS DATA ──────────────────────────────
X = df.drop('label', axis=1)        # Input features (7 columns)
y = df['label']                     # Output crop name

# Encode crop names to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features (normalize)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42
)

print("\n── TRAIN TEST SPLIT ──")
print(f"   Training Records : {X_train.shape[0]}  (80%)")
print(f"   Testing Records  : {X_test.shape[0]}   (20%)")


# ── STEP 6: TRAIN 3 MODELS ───────────────────────────────
print("\n── TRAINING MODELS ──")

# Model 1 — Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print(" Naive Bayes        — Trained")

# Model 2 — Logistic Regression
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)
print(" Logistic Regression — Trained")

# Model 3 — Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print(" Random Forest       — Trained")


# ── STEP 7: EVALUATE ALL 3 MODELS ────────────────────────
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n" + "=" * 60)
print("   MODEL ACCURACY RESULTS")
print("=" * 60)
print(f"   Naive Bayes         : {nb_acc * 100:.2f}%")
print(f"   Logistic Regression : {lr_acc * 100:.2f}%")
print(f"   Random Forest       : {rf_acc * 100:.2f}%  ✅ BEST")
print("=" * 60)

# 7A — Accuracy Bar Chart (All 3 Models)
models     = ['Naive Bayes', 'Logistic\nRegression', 'Random\nForest']
accuracies = [nb_acc * 100, lr_acc * 100, rf_acc * 100]
colors     = ['#f472b6', '#fb923c', '#4ade80']

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=colors,
               edgecolor='black', width=0.5)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(80, 102)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
plt.show()
print("✅ Accuracy chart saved as accuracy_comparison.png")

# 7B — Confusion Matrix — Naive Bayes
plt.figure(figsize=(14, 12))
cm_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix — Naive Bayes', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix_nb.png', dpi=150)
plt.show()
print("✅ NB Confusion Matrix saved as confusion_matrix_nb.png")

# 7C — Confusion Matrix — Random Forest
plt.figure(figsize=(14, 12))
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix — Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=150)
plt.show()
print("✅ RF Confusion Matrix saved as confusion_matrix_rf.png")

# 7D — Classification Report — Random Forest
print("\n── RANDOM FOREST CLASSIFICATION REPORT ──")
print(classification_report(y_test, rf_pred,
                            target_names=le.classes_))


# ── STEP 8: XAI — FEATURE IMPORTANCE ─────────────────────
feature_names  = list(X.columns)
importances    = rf_model.feature_importances_
sorted_idx     = np.argsort(importances)[::-1]
sorted_names   = [feature_names[i] for i in sorted_idx]
sorted_values  = importances[sorted_idx]

print("\n── XAI — FEATURE IMPORTANCE (Random Forest) ──")
for name, val in zip(sorted_names, sorted_values):
    level = "High" if val > 0.2 else "Medium" if val > 0.1 else "Low"
    print(f"   {name:15} : {val:.4f}  → {level} Contribution")

# Feature Importance Bar Chart
colors_xai = ['#ef4444' if v > 0.2 else
               '#f97316' if v > 0.1 else
               '#3b82f6' for v in sorted_values]

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_names, sorted_values,
                color=colors_xai, edgecolor='black')
plt.title('XAI — Feature Importance (Why This Crop?)',
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
for bar, val in zip(bars, sorted_values):
    level = "High" if val > 0.2 else "Medium" if val > 0.1 else "Low"
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f} ({level})', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('xai_feature_importance.png', dpi=150)
plt.show()
print("✅ XAI chart saved as xai_feature_importance.png")


# ── STEP 9: PREDICT NEW INPUT ─────────────────────────────
print("\n" + "=" * 60)
print("   REAL-TIME CROP PREDICTION")
print("=" * 60)

# Take input values from user
print("   Enter the following soil and climate values:\n")
N           = float(input("   Nitrogen     (N)          : "))
P           = float(input("   Phosphorus   (P)          : "))
K           = float(input("   Potassium    (K)          : "))
temperature = float(input("   Temperature  (°C)         : "))
humidity    = float(input("   Humidity     (%)          : "))
ph          = float(input("   pH Level     (0-14)       : "))
rainfall    = float(input("   Rainfall     (mm)         : "))

# Scale the new input same as training data
new_input    = scaler.transform([[N, P, K, temperature,
                                   humidity, ph, rainfall]])
prediction   = rf_model.predict(new_input)
predicted_crop = le.inverse_transform(prediction)[0]

print("\n┌─────────────────────────────────────────────┐")
print("│           ENTER SOIL CONDITIONS            │")
print("├──────────────────┬──────────────────────────┤")
print(f"│ Nitrogen   : {N:<5} │ Phosphorus  : {P:<10}  │")
print(f"│ Potassium  : {K:<5} │ Temperature : {temperature:<5} °C     │")
print(f"│ Humidity   : {humidity:<5} │ pH          : {ph:<5}         │")
print(f"│ Rainfall   : {rainfall:<5} mm                       │")
print("└─────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────┐")
print("│         BEST CROP RECOMMENDATION            │")
print("├─────────────────────────────────────────────┤")
print(f"│  ✅ Recommended Crop : {predicted_crop.upper():<22}│")
print("├─────────────────────────────────────────────┤")
print("│          REASON FOR PREDICTION (XAI)        │")
print("├─────────────────────────────────────────────┤")

bar_full  = "█████████████████████████"   # High
bar_med   = "████████████████"             # Medium
bar_low   = "████████"                     # Low

for name, val in zip(sorted_names, sorted_values):
    if val > 0.2:
        level = "High Contribution  "
        bar   = bar_full
    elif val > 0.1:
        level = "Medium Contribution"
        bar   = bar_med
    else:
        level = "Low Contribution   "
        bar   = bar_low
    print(f"│ {name:<12} {bar:<25} {level} │")

print("└─────────────────────────────────────────────┘")

print("\n" + "=" * 60)
print("   CODE EXECUTION COMPLETE")
print("=" * 60)

# ── SUMMARY OF SAVED FILES ────────────────────────────────
print("\n📁 Files Saved:")
print("   heatmap.png               → Feature Correlation")
print("   histogram.png             → Data Distribution")
print("   crop_count.png            → Records per Crop")
print("   accuracy_comparison.png   → Model Accuracy Chart")
print("   confusion_matrix_nb.png   → Naive Bayes Matrix")
print("   confusion_matrix_rf.png   → Random Forest Matrix")
print("   xai_feature_importance.png→ XAI Explanation")
