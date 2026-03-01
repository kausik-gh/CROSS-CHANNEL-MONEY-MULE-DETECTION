import pandas as pd

from backend.simulation import reset_simulation
from backend.attacks import attack_registry
from backend.features import build_transaction_graph, extract_node_features


# -----------------------------
# Multi-Run Dataset Builder
# -----------------------------

def build_multi_run_dataset(num_runs=20):

    all_features = []
    attack_index = 0

    for _ in range(num_runs):

        accounts_df, transactions_df = reset_simulation()

        attack_function = attack_registry[attack_index % len(attack_registry)]

        accounts_df, transactions_df, attack_name, attack_time = attack_function(
            accounts_df, transactions_df
        )

        attack_transactions = transactions_df[
            transactions_df["timestamp"] >= attack_time
        ]

        G = build_transaction_graph(accounts_df, attack_transactions)

        features_df = extract_node_features(accounts_df, attack_transactions, G)

        all_features.append(features_df)

        attack_index += 1

    final_dataset = pd.concat(all_features, ignore_index=True)

    return final_dataset
