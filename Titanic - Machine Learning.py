
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn convergence/deprecation noise

# Global plot style — whitegrid improves readability; higher DPI for sharper exports
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# STEP 1 — LOAD & EXPLORE THE DATA

print("=" * 60)
print("STEP 1: LOAD & EXPLORE")
print("=" * 60)

df = pd.read_csv("train.csv")  # Load the labeled training data (891 passengers)

print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data types & non-null counts ---")
print(df.info())  # Quickly reveals which columns have missing values

print("\n--- Descriptive statistics ---")
print(df.describe())  # Check for outliers (e.g., extreme Fare values) and value ranges

print("\n--- Missing values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])  # Only show columns that actually have gaps

print("\n--- Survival rate (overall) ---")
# normalize=True gives proportions (0.0–1.0) instead of raw counts
print(df["Survived"].value_counts(normalize=True).rename({0: "Died", 1: "Survived"}))

# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS

print("\n" + "=" * 60)
print("STEP 2: EDA & VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA — Survival Patterns", fontsize=16, fontweight="bold", y=1.01)

# 2.1 Overall Survival Count
ax = axes[0, 0]
# Map 0/1 to labels so the x-axis is human-readable in the chart
df["Survived"].map({0: "Died", 1: "Survived"}).value_counts().plot(
    kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5
)
ax.set_title("Overall Survival Count")
ax.set_xlabel("")
ax.set_ylabel("Passengers")
ax.tick_params(axis="x", rotation=0)
# Annotate each bar with its exact count for quick reading
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height() + 5),
                ha="center", fontsize=11)

# 2.2 Survival by Sex
ax = axes[0, 1]
# Compute mean of Survived (0/1) per group — equivalent to the survival rate
survival_sex = df.groupby("Sex")["Survived"].mean().reset_index()
sns.barplot(data=survival_sex, x="Sex", y="Survived", ax=ax,
            palette={"male": "#3498db", "female": "#e91e8c"})
ax.set_title("Survival Rate by Sex")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", fontsize=11)

# 2.3 Survival by Passenger Class
ax = axes[0, 2]
survival_class = df.groupby("Pclass")["Survived"].mean().reset_index()
# Colors loosely mirror class prestige: gold (1st), silver (2nd), bronze (3rd)
sns.barplot(data=survival_class, x="Pclass", y="Survived", ax=ax,
            palette=["#f1c40f", "#95a5a6", "#cd7f32"])
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Class (1=First, 3=Third)")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", fontsize=11)

# 2.4 Age Distribution by Survival
ax = axes[1, 0]
# Overlay two histograms with alpha transparency to show overlap between groups
df[df["Survived"] == 0]["Age"].dropna().plot(kind="hist", bins=30, alpha=0.6,
                                               color="#e74c3c", label="Died", ax=ax)
df[df["Survived"] == 1]["Age"].dropna().plot(kind="hist", bins=30, alpha=0.6,
                                               color="#2ecc71", label="Survived", ax=ax)
ax.set_title("Age Distribution by Survival")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()

# 2.5 Fare Distribution by Class (boxplot)
ax = axes[1, 1]
# Boxplot reveals median, spread, and outliers per class; cap y-axis to avoid extreme outliers
sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax,
            palette=["#f1c40f", "#95a5a6", "#cd7f32"])
ax.set_title("Fare Distribution by Class")
ax.set_xlabel("Passenger Class")
ax.set_ylim(0, 300)  # A few very high fares exist; cap at 300 to keep the plot readable

# 2.6 Survival by Embarkation Port
ax = axes[1, 2]
survival_emb = df.groupby("Embarked")["Survived"].mean().reset_index()
sns.barplot(data=survival_emb, x="Embarked", y="Survived", ax=ax, palette="pastel")
ax.set_title("Survival Rate by Embarkation Port\n(C=Cherbourg, Q=Queenstown, S=Southampton)")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.02),
                ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("eda_survival_patterns.png", bbox_inches="tight")
plt.show()
print("Saved: eda_survival_patterns.png")

# 2.7 Correlation Heatmap (numeric features only)
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number])  # Exclude object columns (Sex, Embarked, etc.)
corr = numeric_cols.corr()
# Mask the upper triangle to avoid showing duplicate correlation values
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: eda_correlation_heatmap.png")

# STEP 3 — DATA CLEANING & PREPROCESSING
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING & PREPROCESSING")
print("=" * 60)

df_clean = df.copy()  # Work on a copy so the original raw data stays intact for reference

# 3.1 Fill missing Age with median grouped by Sex + Pclass
# Using a group median (rather than the overall median) is more accurate because
# age distributions differ significantly across gender and ticket class.
age_median = df_clean.groupby(["Sex", "Pclass"])["Age"].transform("median")
df_clean["Age"] = df_clean["Age"].fillna(age_median)
print(f"Age missing after fill: {df_clean['Age'].isnull().sum()}")

# 3.2 Fill missing Embarked with mode
# Only 2 rows are missing Embarked, so filling with the most common port (S) is safe.
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])
print(f"Embarked missing after fill: {df_clean['Embarked'].isnull().sum()}")

# 3.3 Drop Cabin (too many missing), Name, Ticket (not useful raw)
# Cabin is ~77% missing — too sparse to impute reliably.
# Ticket numbers are alphanumeric IDs with no direct predictive signal in raw form.
df_clean.drop(columns=["Cabin", "Ticket"], inplace=True)

# 3.4 Feature Engineering
# Extract title from Name — titles encode social status and gender simultaneously,
# making them a more informative signal than Sex alone.
df_clean["Title"] = df_clean["Name"].str.extract(r",\s*([^\.]+)\.")
# Collapse rare/foreign titles into a single "Rare" category to avoid sparse classes
df_clean["Title"] = df_clean["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
     "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
)
# Normalize French titles to their English equivalents
df_clean["Title"] = df_clean["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
print(f"\nTitle value counts:\n{df_clean['Title'].value_counts()}")

df_clean.drop(columns=["Name"], inplace=True)  # Name itself is no longer needed after title extraction

# Family size: total family members on board including the passenger themselves
# Research shows solo travelers and very large families had lower survival rates.
df_clean["FamilySize"] = df_clean["SibSp"] + df_clean["Parch"] + 1
df_clean["IsAlone"] = (df_clean["FamilySize"] == 1).astype(int)  # Binary flag: 1 if travelling alone

# Bin continuous Age into meaningful life-stage categories.
# This lets the model capture non-linear age effects (e.g., children prioritized for lifeboats).
df_clean["AgeGroup"] = pd.cut(df_clean["Age"],
                               bins=[0, 12, 18, 35, 60, 100],
                               labels=["Child", "Teen", "Adult", "Middle-aged", "Senior"])

# 3.5 Encode categorical features
# LabelEncoder converts string categories to integers so sklearn models can process them.
# Note: for tree-based models this is fine; for linear models ordinal encoding can introduce
# unintended ordering, but here it's acceptable given the small number of categories.
le = LabelEncoder()
for col in ["Sex", "Embarked", "Title", "AgeGroup"]:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

print(f"\nCleaned dataset shape: {df_clean.shape}")
print(df_clean.head())
print(f"\nRemaining missing values:\n{df_clean.isnull().sum()}")


# STEP 4 — PREDICTIVE MODELING

print("\n" + "=" * 60)
print("STEP 4: PREDICTIVE MODELING")
print("=" * 60)

# Define the feature set and target variable
# PassengerId is excluded — it's just a row index with no predictive value.
# SibSp and Parch are kept alongside FamilySize so the model can use both
# the raw counts and the derived total.
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone", "AgeGroup"]
TARGET = "Survived"

X = df_clean[FEATURES]
y = df_clean[TARGET]

# Train/test split (80/20, stratified)
# stratify=y ensures both splits preserve the same ~38% survival rate as the full dataset,
# preventing an accidentally skewed test set from misleading the accuracy metric.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# ---- Model 1: Logistic Regression ----
# Serves as the interpretable baseline. max_iter=500 gives the solver enough
# iterations to converge on this dataset (default 100 can be too few).
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"\nLogistic Regression Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, lr_pred, target_names=["Died", "Survived"]))

# ---- Model 2: Random Forest ----
# An ensemble of 200 decision trees. max_depth=6 limits tree complexity to
# reduce overfitting while still capturing non-linear interactions.
# n_jobs=-1 uses all available CPU cores for faster training.
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy:      {rf_acc:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Died", "Survived"]))

# ---- Comparison Plot ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

# Confusion Matrix — Logistic Regression
# Shows true positives, false positives, true negatives, false negatives
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred),
                       display_labels=["Died", "Survived"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"Logistic Regression\nAccuracy: {lr_acc:.1%}")

# Confusion Matrix — Random Forest
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred),
                       display_labels=["Died", "Survived"]).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Random Forest\nAccuracy: {rf_acc:.1%}")

# Feature Importance (Random Forest)
# Importance is measured by mean decrease in Gini impurity across all trees.
# Sorted ascending so the most important feature appears at the top of the horizontal bar chart.
feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
feat_imp.plot(kind="barh", ax=axes[2], color="#3498db", edgecolor="white")
axes[2].set_title("Feature Importance (Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_evaluation.png", bbox_inches="tight")
plt.show()
print("Saved: model_evaluation.png")

# SUMMARY

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"  Dataset:               Titanic (train.csv) — {df.shape[0]} passengers")
print(f"  Features used:         {len(FEATURES)}")
print(f"  Logistic Regression:   {lr_acc:.2%} accuracy")
print(f"  Random Forest:         {rf_acc:.2%} accuracy")
# Pick the better model by accuracy; ties go to Random Forest (more robust)
best = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"  Best model:            {best}")
print("\nTop 3 most important features (Random Forest):")
for feat, score in feat_imp.sort_values(ascending=False).head(3).items():
    print(f"  - {feat}: {score:.4f}")
print("\nAll charts saved as .png in the current directory.")
print("=" * 60)