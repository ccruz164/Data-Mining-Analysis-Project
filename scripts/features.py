"""
Feature building for the Airbnb New User Bookings models.

Follows the same cleaning choices as build_models.py (drop id / date_first_booking,
treat unusual ages as missing, date parts, one-hot categoricals) but:
  - processes train AND test together so the one-hot columns line up
    (needed to make a Kaggle submission)
  - converts ages entered as birth years (e.g. 1952) into real ages
  - adds features summarizing each user's web sessions (sessions.csv)

Run from the repo root. sessions.csv is too big for GitHub, so download it from
Kaggle and put it in data/raw/ (it's already in .gitignore).
"""

import numpy as np
import pandas as pd

RAW = "data/raw/"

CATEGORICAL_COLS = [
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


def load_users():
    """Load train + test users into one table. Test rows have no country_destination."""
    train = pd.read_csv(RAW + "train_users_2.csv")
    test = pd.read_csv(RAW + "test_users.csv")
    train["is_train"] = True
    test["is_train"] = False
    return pd.concat([train, test], ignore_index=True)


def clean_users(users):
    df = users.drop(columns=["date_first_booking"])  # leaks the answer; always empty in test

    # Some users typed their birth year instead of their age (e.g. 1952).
    # The data was collected around 2014-2015, so convert those to an age.
    birth_year = (df["age"] > 1900) & (df["age"] < 2010)
    df.loc[birth_year, "age"] = 2015 - df.loc[birth_year, "age"]

    # Same rule as build_models.py: anything outside 14-99 is treated as missing, then -1
    df.loc[(df["age"] >= 100) | (df["age"] < 14), "age"] = np.nan
    df["age"] = df["age"].fillna(-1)

    # Date features
    dac = pd.to_datetime(df["date_account_created"])
    tfa = pd.to_datetime(df["timestamp_first_active"].astype(str), format="%Y%m%d%H%M%S")
    df["dac_year"] = dac.dt.year
    df["dac_month"] = dac.dt.month
    df["dac_day"] = dac.dt.day
    df["dac_dayofweek"] = dac.dt.dayofweek
    df["tfa_year"] = tfa.dt.year
    df["tfa_month"] = tfa.dt.month
    df["tfa_day"] = tfa.dt.day
    df["tfa_hour"] = tfa.dt.hour
    df["tfa_dayofweek"] = tfa.dt.dayofweek
    # Days between first visiting the site and creating an account
    df["days_active_to_account"] = (dac - tfa.dt.normalize()).dt.days

    df = df.drop(columns=["timestamp_first_active"])  # keep date_account_created for the time split

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("-unknown-").astype(str)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=np.uint8)
    return df


def session_features(min_count=100):
    """
    One row per user summarizing what they did on the site before booking.
      - how many actions, how many distinct actions, devices used
      - time stats (secs_elapsed = seconds between this action and the next one)
      - how many times they did each action / action_type / action_detail / device
    Values that appear fewer than `min_count` times in the whole file are dropped
    to keep the column count reasonable.
    """
    s = pd.read_csv(RAW + "sessions.csv")
    s = s.dropna(subset=["user_id"])
    for col in ["action", "action_type", "action_detail", "device_type"]:
        s[col] = s[col].fillna("missing")

    grouped = s.groupby("user_id")
    feats = pd.DataFrame({
        "sess_n_actions": grouped.size(),
        "sess_n_unique_actions": grouped["action"].nunique(),
        "sess_n_devices": grouped["device_type"].nunique(),
        "sess_secs_total": grouped["secs_elapsed"].sum(),
        "sess_secs_mean": grouped["secs_elapsed"].mean(),
        "sess_secs_median": grouped["secs_elapsed"].median(),
        "sess_secs_max": grouped["secs_elapsed"].max(),
        "sess_secs_std": grouped["secs_elapsed"].std(),
    })

    count_tables = [feats]
    for col in ["action", "action_type", "action_detail", "device_type"]:
        keep = s[col].value_counts()
        keep = keep[keep >= min_count].index
        sub = s[s[col].isin(keep)]
        counts = pd.crosstab(sub["user_id"], sub[col]).astype(np.float32)
        counts.columns = [f"sess_{col}_{c}" for c in counts.columns]
        count_tables.append(counts)

    # Total time spent per action_type (e.g. lots of time on "booking_request")
    secs = s.pivot_table(index="user_id", columns="action_type", values="secs_elapsed",
                         aggfunc="sum").astype(np.float32)
    secs.columns = [f"sess_secs_{c}" for c in secs.columns]
    count_tables.append(secs)

    out = pd.concat(count_tables, axis=1)
    # LightGBM doesn't allow some characters in column names
    out.columns = [c.replace(" ", "_").replace("-", "_").replace("/", "_") for c in out.columns]
    return out


def build_features(use_sessions=True):
    """
    Returns (df, session_cols). df has every user (train + test) with:
      id, is_train, date_account_created, country_destination (NaN for test), and features.
    """
    df = clean_users(load_users())
    session_cols = []
    if use_sessions:
        sess = session_features()
        session_cols = list(sess.columns)
        df = df.merge(sess, how="left", left_on="id", right_index=True)
        df["has_sessions"] = df["sess_n_actions"].notna().astype(np.uint8)
        session_cols.append("has_sessions")
        # Users with no session rows: counts become 0, time stats stay NaN (trees handle NaN)
        count_cols = [c for c in session_cols if c.startswith(("sess_action", "sess_device", "sess_n_"))]
        df[count_cols] = df[count_cols].fillna(0)
    return df, session_cols
