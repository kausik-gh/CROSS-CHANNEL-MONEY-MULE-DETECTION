import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------
# Build Transaction Graph
# -----------------------------

def build_transaction_graph(accounts_df, transactions_df):

    G = nx.DiGraph()

    # Add nodes (accounts)
    for _, row in accounts_df.iterrows():
        G.add_node(
            row["account_id"],
            is_fraud=row["is_fraud"]
        )

    # Add edges (transactions)
    for _, tx in transactions_df.iterrows():
        G.add_edge(
            tx["sender"],
            tx["receiver"],
            amount=tx["amount"],
            timestamp=tx["timestamp"]
        )

    return G


# -----------------------------
# Visualize Graph
# -----------------------------

def visualize_fraud_subgraph(G):

    fraud_nodes = [n for n in G.nodes if G.nodes[n]["is_fraud"] == 1]

    if not fraud_nodes:
        return None

    nodes_to_draw = set(fraud_nodes)

    for fraud in fraud_nodes:
        nodes_to_draw.update(G.predecessors(fraud))
        nodes_to_draw.update(G.successors(fraud))

    subG = G.subgraph(nodes_to_draw)

    fig, ax = plt.subplots(figsize=(8,6))
    pos = nx.spring_layout(subG, seed=42)

    node_colors = []
    for node in subG.nodes():
        if subG.nodes[node]["is_fraud"] == 1:
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    nx.draw(
        subG,
        pos,
        ax=ax,
        with_labels=True,
        node_size=800,
        node_color=node_colors,
        edge_color="gray",
        font_size=8
    )
    return fig


# -----------------------------
# Feature Engineering
# -----------------------------

def extract_node_features(accounts_df, transactions_df, G):

    feature_data = []
    account_lookup = accounts_df.set_index("account_id").to_dict("index")
    device_counts = accounts_df["device_id"].value_counts().to_dict()

    for node in G.nodes():

        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        incoming_edges = G.in_edges(node, data=True)
        outgoing_edges = G.out_edges(node, data=True)

        total_in_amount = sum(edge[2]["amount"] for edge in incoming_edges)
        total_out_amount = sum(edge[2]["amount"] for edge in outgoing_edges)

        # Retention ratio
        if total_in_amount > 0:
            retention_ratio = (total_in_amount - total_out_amount) / total_in_amount
        else:
            retention_ratio = 0

        # Unique neighbors
        neighbors = set([edge[1] for edge in outgoing_edges] +
                        [edge[0] for edge in incoming_edges])
        unique_neighbors = len(neighbors)

        # Unique channels
        node_transactions = transactions_df[
            (transactions_df["sender"] == node) |
            (transactions_df["receiver"] == node)
        ]
        unique_channels = node_transactions["channel"].nunique()

        # Device cluster size
        device_id = account_lookup[node]["device_id"]


        device_cluster_size = device_counts.get(device_id, 1)


        # Transaction count
        transaction_count = node_transactions.shape[0]
        account_age_days = (
            pd.Timestamp.now() - account_lookup[node]["creation_time"]
        ).days

        feature_data.append({
            "account_id": node,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_in_amount": total_in_amount,
            "total_out_amount": total_out_amount,
            "retention_ratio": retention_ratio,
            "unique_neighbors": unique_neighbors,
            "unique_channels": unique_channels,
            "device_cluster_size": device_cluster_size,
            "transaction_count": transaction_count,
            "is_fraud": account_lookup[node]["is_fraud"],
            "account_age_days": account_age_days
        })

    features_df = pd.DataFrame(feature_data)

    return features_df