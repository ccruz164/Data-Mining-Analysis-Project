"""
Model comparison + Kaggle submission for Airbnb New User Bookings.

What it does:
  1. Builds features for train + test (scripts/features.py)
  2. Holds out the most recent 3 months of training users (Apr-Jun 2014) as a
     validation set. The Kaggle test users signed up Jul-Sep 2014, so validating
     on the latest users is a better stand-in than a random split.
  3. Trains a ladder of models and scores each with NDCG@5 (the Kaggle metric)
  4. Retrains the best model on all training users and writes results/submission.csv

Run from the repo root:  python scripts/train_models.py
Outputs go to results/.
"""

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features

RESULTS = "results/"
VALID_START = "2014-04-01"
SEED = 42


# ---------------------------------------------------------------- metric
def ndcg_at_5(y_true, probs):
    """
    Kaggle's metric. For each user we rank the 12 countries by predicted probability
    and look at the top 5. If the true country is ranked 1st the user scores 1.0,
    2nd -> 1/log2(3) = 0.63, 3rd -> 0.5, 4th -> 0.43, 5th -> 0.39, not in top 5 -> 0.
    The final score is the average over users.
    """
    top5 = np.argsort(-probs, axis=1)[:, :5]
    hits = top5 == np.asarray(y_true)[:, None]
    discounts = 1.0 / np.log2(np.arange(2, 7))
    return float((hits * discounts).sum(axis=1).mean())


def top_k_acc(y_true, probs, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == np.asarray(y_true)[:, None]).any(axis=1).mean())


# ---------------------------------------------------------------- models
def constant_baseline(y_train, n_classes, n_rows):
    """Give every user the same ranking: most common countries first. No features at all."""
    freq = np.bincount(y_train, minlength=n_classes).astype(float)
    return np.tile(freq / freq.sum(), (n_rows, 1))


def lgb_params(n_classes):
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": SEED,
    }


def train_lgb(X_tr, y_tr, X_va, y_va, n_classes):
    """Gradient boosting with early stopping: stop adding trees once validation loss stops improving."""
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    model = lgb.train(
        lgb_params(n_classes), dtrain, num_boost_round=2000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


# ---------------------------------------------------------------- main
def main():
    os.makedirs(RESULTS, exist_ok=True)

    print("Building features (sessions.csv takes a minute)...")
    df, session_cols = build_features(use_sessions=True)

    le = LabelEncoder()
    train = df[df["is_train"]].copy()
    test = df[~df["is_train"]].copy()
    y_all = le.fit_transform(train["country_destination"])
    classes = le.classes_
    n_classes = len(classes)

    # dac_year / tfa_year are dropped as features: every test user signed up in 2014,
    # so the year can't tell test users apart and the model would just be extrapolating.
    # (Validation NDCG@5 went from 0.8527 to 0.8535 without them.)
    drop = ["id", "is_train", "date_account_created", "country_destination", "dac_year", "tfa_year"]
    all_cols = [c for c in df.columns if c not in drop]
    base_cols = [c for c in all_cols if c not in session_cols]  # user table only
    onehot_cols = [c for c in base_cols if df[c].dtype == np.uint8]  # for Naive Bayes

    is_valid = (train["date_account_created"] >= VALID_START).values
    y_tr, y_va = y_all[~is_valid], y_all[is_valid]
    print(f"Train rows: {(~is_valid).sum():,}  Validation rows (>= {VALID_START}): {is_valid.sum():,}  "
          f"Test rows: {len(test):,}")
    print(f"Features: {len(base_cols)} from user table + {len(session_cols)} from sessions")

    def split(cols):
        X = train[cols].astype(np.float32)
        return X[~is_valid], X[is_valid]

    results = []
    preds = {}

    def record(name, features, probs, seconds):
        row = {
            "model": name,
            "features": features,
            "NDCG@5": round(ndcg_at_5(y_va, probs), 4),
            "top1_acc": round(top_k_acc(y_va, probs, 1), 4),
            "top5_acc": round(top_k_acc(y_va, probs, 5), 4),
            "train_seconds": round(seconds, 1),
        }
        results.append(row)
        preds[name] = probs
        print(row)

    # 1. Constant baseline
    t = time.time()
    record("Constant ranking (no model)", "none",
           constant_baseline(y_tr, n_classes, len(y_va)), time.time() - t)

    # 2. Naive Bayes on the one-hot (yes/no) attributes
    Xtr, Xva = split(onehot_cols)
    t = time.time()
    nb = BernoulliNB().fit(Xtr, y_tr)
    record("Naive Bayes (Bernoulli)", "user table, one-hot only", nb.predict_proba(Xva), time.time() - t)

    # 3. Logistic regression
    Xtr, Xva = split(base_cols)
    t = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    lr.fit(Xtr, y_tr)
    record("Logistic regression", "user table", lr.predict_proba(Xva), time.time() - t)

    # 4. Random forest - cc's settings, with and without class balancing
    for cw, label in [("balanced", "Random forest (balanced classes)"), (None, "Random forest")]:
        t = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=cw,
                                    random_state=SEED, n_jobs=-1)
        rf.fit(Xtr, y_tr)
        record(label, "user table", rf.predict_proba(Xva), time.time() - t)

    # 5. LightGBM without sessions
    t = time.time()
    m = train_lgb(Xtr, y_tr, Xva, y_va, n_classes)
    record("LightGBM", "user table", m.predict(Xva, num_iteration=m.best_iteration), time.time() - t)

    # 6. LightGBM with sessions
    Xtr, Xva = split(all_cols)
    t = time.time()
    best = train_lgb(Xtr, y_tr, Xva, y_va, n_classes)
    best_probs = best.predict(Xva, num_iteration=best.best_iteration)
    record("LightGBM", "user table + sessions", best_probs, time.time() - t)

    pd.DataFrame(results).to_csv(RESULTS + "model_comparison.csv", index=False)

    # Per-country breakdown for the best model: how often is the true country in the top 5?
    top5 = np.argsort(-best_probs, axis=1)[:, :5]
    hits = (top5 == y_va[:, None]).any(axis=1)
    per_class = pd.DataFrame({
        "country": classes,
        "support": [int((y_va == i).sum()) for i in range(n_classes)],
        "recall@5": [round(hits[y_va == i].mean(), 4) if (y_va == i).any() else np.nan
                     for i in range(n_classes)],
        "share_ranked_first": [round((top5[:, 0] == i).mean(), 4) for i in range(n_classes)],
    })
    per_class.to_csv(RESULTS + "per_country_best_model.csv", index=False)
    print(per_class)

    importance = pd.DataFrame({
        "feature": all_cols,
        "gain": best.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(RESULTS + "feature_importance.csv", index=False)
    print(importance.head(20))

    # ------------------------------------------------ final model + submission
    # Retrain on ALL labeled users (including the validation months, which are the
    # closest to the test period). Use ~10% more trees since there's more data.
    n_rounds = int(best.best_iteration * 1.1)
    print(f"Retraining on all {len(train):,} users for {n_rounds} rounds...")
    final = lgb.train(lgb_params(n_classes), lgb.Dataset(train[all_cols].astype(np.float32), y_all),
                      num_boost_round=n_rounds)
    test_probs = final.predict(test[all_cols].astype(np.float32))

    # Kaggle format: 5 rows per user, best guess first
    top5 = np.argsort(-test_probs, axis=1)[:, :5]
    ids = np.repeat(test["id"].values, 5)
    countries = classes[top5.ravel()]
    pd.DataFrame({"id": ids, "country": countries}).to_csv(RESULTS + "submission.csv", index=False)
    print("Wrote results/submission.csv")


if __name__ == "__main__":
    main()
