import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import time
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score
from backend.training import build_multi_run_dataset
from backend.controller import simulate_coordinated_attack
from backend.features import (
    build_transaction_graph,
    visualize_fraud_subgraph,
    extract_node_features
)
from backend.detection import (
    rule_based_detection,
    ml_predict,
    ml_detection,
    explain_risk_categories,
    classify_fraud_roles,
    behavioral_drift_detection,
    early_stage_detection,
    adaptive_threshold_update
)
from backend.risk_memory import (
    extract_cluster_signature,
    store_signature,
    compare_signature
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Money Mule Detection",
    layout="wide"
)

st.title("🚨 Cross-Channel Money Mule Detection System")

st.markdown("""
This system simulates coordinated money mule attacks across channels
and uses ML to detect suspicious accounts in real time.
""")

# -----------------------------
# Initialize Session State
# -----------------------------
if "threshold_history" not in st.session_state:
    st.session_state.threshold_history = []

if "fraud_history" not in st.session_state:
    st.session_state.fraud_history = []

if "role_history" not in st.session_state:
    st.session_state.role_history = []

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.45

if "model" not in st.session_state:
    st.session_state.model = None

if "attack_index" not in st.session_state:
    st.session_state.attack_index = 0



# -----------------------------
# Train Model (Only Once)
# -----------------------------
if st.session_state.model is None:

    st.info("Training ML model... Please wait.")

    final_dataset = build_multi_run_dataset(num_runs=10)

    model, cm, report, feature_importances, explainer = ml_detection(final_dataset)

    st.session_state.model = model
    st.session_state.explainer = explainer
    st.session_state.attack_index = 0

    # -----------------------------
    # Create Clean Baseline (No Attack)
    # -----------------------------
    from backend.simulation import reset_simulation

    clean_accounts, clean_transactions = reset_simulation()

    baseline_G = build_transaction_graph(clean_accounts, clean_transactions)
    baseline_features = extract_node_features(
        clean_accounts,
        clean_transactions,
        baseline_G
    )

    st.session_state.baseline_features = baseline_features



model = st.session_state.model

# -----------------------------
# Simulate Attack Button
# -----------------------------
if st.button("🚀 Simulate Coordinated Attack"):

    accounts_df, transactions_df, attack_time, attack_name = simulate_coordinated_attack(
        st.session_state.attack_index
    )
    baseline_features = st.session_state.baseline_features

    if accounts_df is not None:

        # Increase index for next click
        st.session_state.attack_index += 1

        # Filter attack transactions
        attack_transactions = transactions_df[
            transactions_df["timestamp"] >= attack_time
        ]

        # -----------------------------
        # Build Graph
        # -----------------------------
        G = build_transaction_graph(accounts_df, attack_transactions)

        fig = visualize_fraud_subgraph(G)
        if fig:
            st.subheader("📊 Fraud Subgraph View")
            st.pyplot(fig)

        # -----------------------------
        # Feature Extraction
        # -----------------------------
        features_df = extract_node_features(
            accounts_df,
            attack_transactions,
            G
        )
        feature_columns = [
            "in_degree",
            "out_degree",
            "total_in_amount",
            "total_out_amount",
            "retention_ratio",
            "unique_neighbors",
            "unique_channels",
            "device_cluster_size",
            "transaction_count",
            "account_age_days"
        ]

        baseline_features = st.session_state.baseline_features
        drift_df = behavioral_drift_detection(
            baseline_features,
            features_df,
            feature_columns,
            threshold=0.4
        )

        if not drift_df.empty:
            st.subheader("📉 Behavioral Drift Detected")
            st.dataframe(
                drift_df.sort_values("drift_score", ascending=False)
            )
        else:
            st.info("No significant behavioral drift detected.")

        # -----------------------------
        # Rule-Based Detection
        # -----------------------------
        risk_df = rule_based_detection(features_df)

        st.subheader("📌 Rule-Based Risk Scoring")
        st.dataframe(
            risk_df.sort_values("risk_score", ascending=False).head(10)
        )

        # -----------------------------
        # ⚡ Early Stage Detection
        # -----------------------------
        early_df = early_stage_detection(features_df)

        if not early_df.empty:
            st.subheader("⚡ Early Stage Mule Detection")
            st.dataframe(early_df)
        else:
            st.info("No early-stage suspicious accounts detected.")

        # -----------------------------
        # ML Detection
        # -----------------------------
        predictions = ml_predict(
            model,
            features_df,
            threshold=st.session_state.threshold
            )

        st.subheader("🤖 ML Detection Results")
        st.dataframe(
            predictions.sort_values("prediction_score", ascending=False).head(10)
        )

        def risk_severity(score):
            if score > 0.15:
                return "🔴 High"
            elif score > 0.07:
                return "🟠 Medium"
            else:
                return "🟢 Low"
        
        st.markdown(f"### 🎛 Current Detection Threshold: {round(st.session_state.threshold,2)}")
        # =============================
        # 🔄 LIVE ADAPTIVE THRESHOLD
        # =============================

        from sklearn.metrics import precision_score, recall_score
        from backend.detection import adaptive_threshold_update

        true_labels = features_df["is_fraud"]
        pred_labels = predictions["predicted_label"]

        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)

        report_live = {
            "1": {
                "precision": precision,
                "recall": recall
            }
        }

        new_threshold = adaptive_threshold_update(
            st.session_state.threshold,
            report_live
        )

        st.session_state.threshold = new_threshold

        st.markdown("### ⚙ Adaptive Threshold System")
        st.write(f"Current Threshold: {round(st.session_state.threshold, 2)}")
        st.write(f"Live Precision: {round(precision, 2)}")
        st.write(f"Live Recall: {round(recall, 2)}")
        

        # -----------------------------
        # 🔍 Explain Fraud Decision
        # -----------------------------
        st.markdown("## 🔍 Fraud Investigation Panel")

        fraud_predictions = predictions[predictions["predicted_label"] == 1]

        if not fraud_predictions.empty:

            selected_account = st.selectbox(
                "Select fraud account to investigate:",
                fraud_predictions["account_id"].values
            )

            # -----------------------------
            # 🚨 Fraud Alert Header
            # -----------------------------
            prob_row = predictions[predictions["account_id"] == selected_account]
            if not prob_row.empty:
                    prob = prob_row["prediction_score"].values[0]
            else:
                prob = 0


            st.markdown("### 🚨 Fraud Alert")
            st.markdown(f"**Account:** {selected_account}")
            st.markdown(f"**Risk Score:** {round(prob*100,2)}%")

            if prob > 0.75:
                st.error("🔴 High Confidence Fraud")
            elif prob > 0.5:
                st.warning("🟠 Medium Risk Fraud")
            else:
                st.info("🟢 Low Confidence")

            # -----------------------------
            # 📊 SHAP Technical Breakdown
            # -----------------------------
            feature_columns = model.feature_list


            account_features = features_df[
                features_df["account_id"] == selected_account
            ][feature_columns]

            # -----------------------------
            # 🧩 Fraud Role Classification
            # -----------------------------

            # Classify roles for all accounts
            roles_df = classify_fraud_roles(features_df)

            # Filter only predicted fraud accounts
            fraud_roles = roles_df[
            roles_df["account_id"].isin(
                    fraud_predictions["account_id"]
                )
            ]

            # Get selected account role
            selected_role = fraud_roles[
                fraud_roles["account_id"] == selected_account
            ]["role"].values

            st.subheader("🧩 Fraud Role Classification")

            if len(selected_role) > 0:
                st.markdown(f"**Assigned Role:** {selected_role[0]}")
            else:
                st.markdown("Role could not be determined.")

            # Show cluster role distribution
            st.markdown("### Fraud Ecosystem Role Distribution")
            st.dataframe(
                fraud_roles["role"].value_counts().reset_index().rename(
                    columns={"index": "Role", "role": "Count"}
                )
            )
            shap_values = st.session_state.explainer.shap_values(account_features)

            # Handle binary classifier output
            if isinstance(shap_values, list):
                shap_array = shap_values[1]   # fraud class
            else:
                shap_array = shap_values

            # Convert to numpy
            shap_array = np.array(shap_array)

            # If 3D (1, features, classes) → take class 1
            if shap_array.ndim == 3:
                shap_array = shap_array[0][:, 1]

            # If 2D (1, features) → flatten
            elif shap_array.ndim == 2:
                shap_array = shap_array[0]

            # Force final 1D
            shap_array = np.array(shap_array).flatten()

            # Ensure correct length
            if len(shap_array) != len(feature_columns):
                st.error(f"SHAP mismatch: {len(shap_array)} vs {len(feature_columns)}")
            else:
                # -----------------------------
                # 📊 Risk Category Breakdown
                # -----------------------------
                category_scores = explain_risk_categories(
                    shap_array,
                    feature_columns
                )

                st.subheader("📊 Risk Contribution Breakdown")

                sorted_categories = sorted(
                    category_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                risk_df = pd.DataFrame(
                    [(k, round(v * 100, 2)) for k, v in sorted_categories],
                    columns=["Risk Category", "Contribution (%)"]
                )

                st.dataframe(risk_df)

                # -----------------------------
                # 🧠 Human Explanation Section
                # -----------------------------
                st.subheader("🧠 Why Was This Flagged?")

                explanations = []

                if category_scores.get("Velocity Risk", 0) > 0.1:
                    explanations.append("• Rapid movement of funds between accounts")

                if category_scores.get("Shared Device Risk", 0) > 0.05:
                    explanations.append("• Linked to multiple accounts via same device")

                if category_scores.get("Ring Participation Risk", 0) > 0.1:
                    explanations.append("• Participated in suspicious transaction ring")

                if category_scores.get("Retention Risk", 0) > 0.05:
                    explanations.append("• Funds quickly transferred out after receiving")

                if category_scores.get("Channel Risk", 0) > 0.05:
                    explanations.append("• Unusual multi-channel activity detected")

                if explanations:
                    for exp in explanations:
                        st.write(exp)
                else:
                    st.write("No dominant behavioral anomaly detected.")

                # -----------------------------
                # 🔬 Advanced Technical View
                # -----------------------------
                with st.expander("🔬 Advanced Technical SHAP Breakdown"):
                    shap_df = pd.DataFrame({
                        "feature": feature_columns,
                        "impact": shap_array
                    }).sort_values(
                        "impact",
                        key=lambda x: abs(x),
                        ascending=False
                    )

                    st.dataframe(shap_df)

        else:
            st.info("No fraud predicted in this simulation.")
        
        # -----------------------------
        # Fraud Role Classification
        # -----------------------------
        predicted_fraud_ids = predictions[
        predictions["predicted_label"] == 1
        ]["account_id"].values

        fraud_only_features = features_df[
            features_df["account_id"].isin(predicted_fraud_ids)
        ]


        if not fraud_only_features.empty:

            roles_df = classify_fraud_roles(fraud_only_features)

            st.subheader("🎭 Fraud Role Classification")
            st.dataframe(roles_df)

        else:
            st.info("No fraud accounts detected for role classification.")

        # -----------------------------
        # 🧠 Fraud Pattern Memory System
        # -----------------------------
        signature = extract_cluster_signature(G, features_df)

        if signature:

            similarity = compare_signature(signature)

            st.subheader("🧠 Fraud Pattern Memory Check")
            st.write(f"Similarity with past fraud patterns: {round(similarity*100,2)}%")

            if similarity > 0.7:
                st.warning("⚠️ High similarity to previous mule ring pattern")
            elif similarity > 0.4:
                st.info("Moderate similarity with stored fraud template")
            else:
                st.success("New structural fraud pattern detected")

            store_signature(signature)


        # -----------------------------
        # Dramatic Reveal
        # -----------------------------
        st.markdown("---")

        with st.spinner("Analyzing behavioral graph patterns..."):
            time.sleep(2)

        st.subheader("🎭 Attack Type Revealed")
        st.success(f"The system detected pattern: **{attack_name}**")

        # -----------------------------
        # Detection Summary
        # -----------------------------
        true_fraud = accounts_df[accounts_df["is_fraud"] == 1]["account_id"].values
        predicted_fraud = predictions[
            predictions["predicted_label"] == 1
        ]["account_id"].values

        correct = set(true_fraud).intersection(set(predicted_fraud))

        st.markdown("### ✅ Detection Summary")
        st.write(f"Fraud Accounts Injected: {len(true_fraud)}")
        st.write(f"Fraud Accounts Detected: {len(correct)}")

        if len(correct) == len(true_fraud):
            st.success("All fraud accounts successfully detected.")
        else:
            st.warning("Some fraud accounts were missed.")
        
        # Track threshold history
        st.session_state.threshold_history.append(st.session_state.threshold)

        # Track fraud detection rate
        fraud_rate = len(correct) / len(true_fraud) if len(true_fraud) > 0 else 0
        st.session_state.fraud_history.append(fraud_rate)

        # Track roles
        if not fraud_only_features.empty:
            role_counts = roles_df["role"].value_counts().to_dict()
            st.session_state.role_history.append(role_counts)
        
        # =====================================
        # STEP 3 — SYSTEM INTELLIGENCE DASHBOARD
        # =====================================

        st.markdown("---")
        st.subheader("🧠 System Intelligence Dashboard")

        import matplotlib.pyplot as plt

        # -----------------------------
        # Threshold Trend
        # -----------------------------
        if st.session_state.threshold_history:
            fig1, ax1 = plt.subplots()
            ax1.plot(st.session_state.threshold_history)
            ax1.set_title("Adaptive Threshold Trend")
            ax1.set_ylabel("Threshold")
            ax1.set_xlabel("Simulation Run")
            st.pyplot(fig1)

        # -----------------------------
        # Fraud Detection Trend
        # -----------------------------
        if st.session_state.fraud_history:
            fig2, ax2 = plt.subplots()
            ax2.plot(st.session_state.fraud_history)
            ax2.set_title("Fraud Detection Success Rate")
            ax2.set_ylabel("Detection Rate")
            ax2.set_xlabel("Simulation Run")
            st.pyplot(fig2)

        # -----------------------------
        # Role Distribution (Latest Run)
        # -----------------------------
        if st.session_state.role_history:
            latest_roles = st.session_state.role_history[-1]

            fig3, ax3 = plt.subplots()
            ax3.bar(latest_roles.keys(), latest_roles.values())
            ax3.set_title("Fraud Role Distribution")
            ax3.set_xticklabels(latest_roles.keys(), rotation=45)
            st.pyplot(fig3)

    else:
        st.warning("All attack patterns executed.")


