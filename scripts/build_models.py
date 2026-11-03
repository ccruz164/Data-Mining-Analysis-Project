import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/train_users_2.csv')

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

# Split the dataset into training and validation sets
# The test set does not have labels so this is useful for measuring precision and recall of the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")

print(df.head())

rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%\n")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=[str(cls) for cls in le.classes_],
        zero_division=0,
    )
)