import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/raw/train_users_2.csv')

# Drop to prevent leakage of information from the target variable
# An existing value for date_first_booking would indicate that the user has already booked, which is what we are trying to predict
# ID is also not a feature
df = df.drop(columns=["id", "date_first_booking"])

# Handle missing values in the "age" column
# Treat unordinary ages as missing values
df.loc[(df["age"] >= 100) | (df["age"] < 14), "age"] = np.nan

# Exact count of missing values
missing_count = df['age'].isna().sum()

# Exact percentage of the total dataset
missing_pct = (missing_count / len(df)) * 100

print(f"Missing age count: {missing_count:,}")
print(f"Missing age percentage: {missing_pct:.2f}%")
# Assign missing values a -1 for unknown
df["age"] = df["age"].fillna(-1)
print(df["age"].describe())

# Transform date columns into datetime objects and create relevant features
df["date_account_created"] = pd.to_datetime(df["date_account_created"])
df["dac_year"] = df["date_account_created"].dt.year
df["dac_month"] = df["date_account_created"].dt.month
df["dac_day"] = df["date_account_created"].dt.day
df["dac_dayofweek"] = df["date_account_created"].dt.dayofweek

df["tfa"] = pd.to_datetime(
    df["timestamp_first_active"].astype(str), format="%Y%m%d%H%M%S"
)
df["tfa_year"] = df["tfa"].dt.year
df["tfa_month"] = df["tfa"].dt.month
df["tfa_day"] = df["tfa"].dt.day
df["tfa_hour"] = df["tfa"].dt.hour
df["tfa_dayofweek"] = df["tfa"].dt.dayofweek

df = df.drop(columns=["date_account_created", "timestamp_first_active", "tfa"])

# Separate target variable
le = LabelEncoder()
y = le.fit_transform(df["country_destination"])
X = df.drop(columns=["country_destination"])

# One hot encoding for categorical features
categorical_cols = [
    "gender",
    "signup_method",
    "signup_flow",
    "language",
    "affiliate_channel",
    "affiliate_provider",
    "first_affiliate_tracked",
    "signup_app",
    "first_device_type",
    "first_browser",
]
X[categorical_cols] = X[categorical_cols].fillna("-unknown-").astype(str)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X.to_csv("data/processed/cleaned_test.csv", index=False)

# Split the dataset into training and validation sets
# The test set does not have labels so this is useful for measuring precision and recall of the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")

print(df.head())

# Note for report: Top 5 Accuracy is much higher when class weight is not set to balanced, but not actually better at predicting the minority classes
# NDF has the most data and the model can get a 95% top 5 accuracy by predicting NDF for most users, but this is not useful for the business problem of predicting the minority classes
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_probs = rf.predict_proba(X_val)
top_5_accuracy = top_k_accuracy_score(y_val, y_probs, k=5)
print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")

top_5_preds = np.argsort(y_probs, axis=1)[:, -5:]
hits = np.any(top_5_preds == y_val[:, None], axis=1)

# Calculate Recall@5 for each destination class
classes = np.unique(y_val)
class_recall_at_5 = {}

for cls_idx in classes:
    class_mask = y_val == cls_idx
    actual_count = np.sum(class_mask)
    hits_in_class = np.sum(hits[class_mask])
    recall_at_5 = hits_in_class / actual_count if actual_count > 0 else 0
    class_recall_at_5[le.classes_[cls_idx]] = {
        "Support": actual_count,
        "Hits@5": hits_in_class,
        "Recall@5": f"{recall_at_5 * 100:.2f}%",
    }

recall_df = pd.DataFrame(class_recall_at_5).T
print(recall_df)