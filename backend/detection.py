import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import numpy as np

# -----------------------------
# Rule-Based Fraud Detection
# -----------------------------

def rule_based_detection(features_df):

    risk_results = []

    for _, row in features_df.iterrows():

        score = 0
        reasons = []

        # Fan-In
        if row["in_degree"] >= 2:
            score += 2
            reasons.append("High In-Degree")

        # Fan-Out
        if row["out_degree"] >= 2:
            score += 2
            reasons.append("High Out-Degree")

        # Money Passing Through
        if row["retention_ratio"] < 0.2 and row["total_in_amount"] > 0:
            score += 2
            reasons.append("Low Retention Ratio")

        # Shared Device
        if row["device_cluster_size"] > 2:
            score += 3
            reasons.append("Shared Device Cluster")

        # Channel Burst
        if row["unique_channels"] > 2:
            score += 2
            reasons.append("High Channel Diversity")

        # High Transaction Volume
        if row["transaction_count"] >= 3:
            score += 1
            reasons.append("High Transaction Count")

        risk_results.append({
            "account_id": row["account_id"],
            "risk_score": score,
            "reasons": reasons,
            "is_fraud": row["is_fraud"]
        })

    return pd.DataFrame(risk_results)

# -----------------------------
# ML-Based Fraud Detection
# -----------------------------

def ml_detection(features_df):

    feature_columns = [
        "in_degree",
        "out_degree",
        "total_in_amount",
        "total_out_amount",
        "retention_ratio",
        "unique_neighbors",
        "unique_channels",
        "device_cluster_size",
        "transaction_count"
    ]

    X = features_df[feature_columns]
    y = features_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
         class_weight="balanced"
    )

    model.fit(X_train, y_train)
    model.feature_list = feature_columns

    feature_importances = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_}).sort_values("importance", ascending=False)


    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True,zero_division=0)

    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)


    # 🔥 SHAP explainer
    explainer = shap.TreeExplainer(model)

    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, cm, report, feature_importances, explainer
# -----------------------------
# ML Prediction (Inference)
# -----------------------------

def ml_predict(model, features_df, threshold=0.5):

    feature_columns = [
        "in_degree",
        "out_degree",
        "total_in_amount",
        "total_out_amount",
        "retention_ratio",
        "unique_neighbors",
        "unique_channels",
        "device_cluster_size",
        "transaction_count"
    ]

    X = features_df[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]

    results = features_df[["account_id", "is_fraud"]].copy()
    results["prediction_score"] = probabilities
    results["predicted_label"] = (probabilities > threshold).astype(int)

    return results

def explain_risk_categories(shap_values, feature_columns):

    category_map = {
        "Velocity Risk": [
            "transaction_count",
            "out_degree"
        ],
        "Shared Device Risk": [
            "device_cluster_size"
        ],
        "Ring Participation Risk": [
            "unique_neighbors",
            "in_degree",
            "out_degree"
        ],
        "Channel Risk": [
            "unique_channels"
        ],
        "Retention Risk": [
            "retention_ratio"
        ]
    }

    category_scores = {}

    for category, features in category_map.items():

        total_impact = 0

        for feature in features:
            if feature in feature_columns:
                idx = feature_columns.index(feature)

                value = shap_values[idx]

                # Force scalar (handle array cases safely)
                if isinstance(value, (list, np.ndarray)):
                    value = float(np.array(value).flatten()[0])

                total_impact += abs(float(value))

        category_scores[category] = round(float(np.array(total_impact).flatten()[0]), 4)

    return category_scores

# -----------------------------
# Fraud Role Classification
# -----------------------------
def classify_fraud_roles(features_df):

    roles = []

    for _, row in features_df.iterrows():

        role = "Unclassified"

        # 1️⃣ Ring Coordinator (strongest structural pattern)
        if (
            row["in_degree"] >= 2 and
            row["out_degree"] >= 2 and
            row["unique_neighbors"] >= 3
        ):
            role = "Ring Coordinator"

        # 2️⃣ Distributor Mule
        elif (
            row["out_degree"] > row["in_degree"] and
            row["retention_ratio"] < 0.3
        ):
            role = "Distributor Mule"

        # 3️⃣ Collector Mule
        elif (
            row["in_degree"] > row["out_degree"] and
            row["retention_ratio"] < 0.3
        ):
            role = "Collector Mule"

        # 4️⃣ Entry Node
        elif row["in_degree"] == 0 and row["out_degree"] > 0:
            role = "Entry Node"

        # 5️⃣ Exit Node
        elif row["in_degree"] > 0 and row["retention_ratio"] > 0.6:
            role = "Exit Node"

        roles.append({
            "account_id": row["account_id"],
            "role": role
        })

    return pd.DataFrame(roles)

def early_stage_detection(features_df):

    early_flags = []

    for _, row in features_df.iterrows():

        risk = 0

        # Very new account
        if row["account_age_days"] <= 5:
            risk += 1

        # High transaction count quickly
        if row["transaction_count"] >= 3:
            risk += 1

        # High outgoing velocity
        if row["out_degree"] >= 2:
            risk += 1

        if risk >= 2:
            early_flags.append({
                "account_id": row["account_id"],
                "early_risk_score": risk
            })

    return pd.DataFrame(early_flags)

def behavioral_drift_detection(
    baseline_features,
    current_features,
    feature_columns,
    threshold=0.5
):

    drift_results = []

    baseline_lookup = baseline_features.set_index("account_id")
    current_lookup = current_features.set_index("account_id")

    common_accounts = set(baseline_lookup.index).intersection(
        set(current_lookup.index)
    )

    for acc in common_accounts:

        baseline_vector = baseline_lookup.loc[acc][feature_columns].values
        current_vector = current_lookup.loc[acc][feature_columns].values

        # Normalize vectors
        baseline_vector = np.array(baseline_vector, dtype=float)
        current_vector = np.array(current_vector, dtype=float)

        # Euclidean distance
        drift_score = np.linalg.norm(current_vector - baseline_vector)

        if drift_score > threshold:
            drift_results.append({
                "account_id": acc,
                "drift_score": drift_score
            })

    return pd.DataFrame(drift_results)

def adaptive_threshold_update(current_threshold, report):

    fraud_recall = report["1"]["recall"]
    fraud_precision = report["1"]["precision"]

    # If recall is low → lower threshold (catch more fraud)
    if fraud_recall < 0.7:
        current_threshold -= 0.1

    # If precision is low → increase threshold (reduce false positives)
    if fraud_precision < 0.7:
        current_threshold += 0.05

    # Clamp between 0.3 and 0.8
    current_threshold = max(0.3, min(0.8, current_threshold))

    return current_threshold