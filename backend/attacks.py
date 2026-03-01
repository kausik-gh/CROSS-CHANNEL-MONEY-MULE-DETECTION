import random
import pandas as pd
from datetime import datetime, timedelta
from backend.generator import START_DATE, CHANNELS

# -----------------------------
# Fraud Pattern 1: Fan-In Attack
# -----------------------------

def fan_in_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Fan-In Pattern")

    mule_account = accounts_df.sample(1).iloc[0]
    mule_id = mule_account["account_id"]

    feeders = accounts_df.sample(5)

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=8)

    for i, feeder in feeders.iterrows():

        if feeder["account_id"] == mule_id:
            continue

        amount = random.randint(20000, 50000)

        if feeder["balance"] < amount:
            continue

        timestamp = base_time + timedelta(minutes=random.randint(0, 5))

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": feeder["account_id"],
            "receiver": mule_id,
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": feeder["device_id"],
            "ip_address": feeder["ip_address"]
        }

        attack_transactions.append(transaction)

        # Update balances
        accounts_df.loc[
            accounts_df["account_id"] == feeder["account_id"],
            "balance"
        ] -= amount

        accounts_df.loc[
            accounts_df["account_id"] == mule_id,
            "balance"
        ] += amount

        transaction_id_start += 1

    # Mule moves out 90%
    outgoing_receiver = accounts_df.sample(1).iloc[0]

    mule_balance = accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "balance"
    ].values[0]

    out_amount = int(mule_balance * 0.9)

    timestamp = base_time + timedelta(minutes=6)

    transaction = {
        "transaction_id": f"T{transaction_id_start}",
        "sender": mule_id,
        "receiver": outgoing_receiver["account_id"],
        "amount": out_amount,
        "timestamp": timestamp,
        "channel": random.choice(CHANNELS),
        "device_id": mule_account["device_id"],
        "ip_address": mule_account["ip_address"]
    }

    attack_transactions.append(transaction)

    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "balance"
    ] -= out_amount

    accounts_df.loc[
        accounts_df["account_id"] == outgoing_receiver["account_id"],
        "balance"
    ] += out_amount

    # Mark mule
    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)

    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Fan-In Pattern", base_time

# -----------------------------
# Fraud Pattern 2: Fan-Out Attack
# -----------------------------

def fan_out_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Fan-Out Pattern")

    mule_account = accounts_df.sample(1).iloc[0]
    mule_id = mule_account["account_id"]

    # Give mule some extra balance to distribute
    extra_amount = random.randint(100000, 200000)

    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "balance"
    ] += extra_amount

    recipients = accounts_df.sample(5)

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=9)

    for i, recipient in recipients.iterrows():

        if recipient["account_id"] == mule_id:
            continue

        amount = random.randint(20000, 50000)

        mule_balance = accounts_df.loc[
            accounts_df["account_id"] == mule_id,
            "balance"
        ].values[0]

        if mule_balance < amount:
            continue

        timestamp = base_time + timedelta(minutes=random.randint(0, 5))

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": mule_id,
            "receiver": recipient["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": mule_account["device_id"],
            "ip_address": mule_account["ip_address"]
        }

        attack_transactions.append(transaction)

        # Update balances
        accounts_df.loc[
            accounts_df["account_id"] == mule_id,
            "balance"
        ] -= amount

        accounts_df.loc[
            accounts_df["account_id"] == recipient["account_id"],
            "balance"
        ] += amount

        transaction_id_start += 1

    # Mark mule
    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)

    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Fan-Out Pattern", base_time

# -----------------------------
# Fraud Pattern 3: Circular Ring
# -----------------------------

def circular_ring_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Circular Transaction Ring")

    # Select 4 random accounts
    ring_accounts = accounts_df.sample(4)
    ring_ids = list(ring_accounts["account_id"])

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=10)

    amount = random.randint(20000, 40000)

    # Ensure all have enough balance
    for acc_id in ring_ids:
        balance = accounts_df.loc[
            accounts_df["account_id"] == acc_id,
            "balance"
        ].values[0]

        if balance < amount:
            accounts_df.loc[
                accounts_df["account_id"] == acc_id,
                "balance"
            ] += amount

    # Create circular transfers
    for i in range(len(ring_ids)):

        sender = ring_ids[i]
        receiver = ring_ids[(i + 1) % len(ring_ids)]

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": accounts_df.loc[
                accounts_df["account_id"] == sender,
                "device_id"
            ].values[0],
            "ip_address": accounts_df.loc[
                accounts_df["account_id"] == sender,
                "ip_address"
            ].values[0]
        }

        attack_transactions.append(transaction)

        # Update balances
        accounts_df.loc[
            accounts_df["account_id"] == sender,
            "balance"
        ] -= amount

        accounts_df.loc[
            accounts_df["account_id"] == receiver,
            "balance"
        ] += amount

        transaction_id_start += 1

    # Mark all ring accounts as fraud
    accounts_df.loc[
        accounts_df["account_id"].isin(ring_ids),
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)

    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Circular Transaction Ring", base_time

# -----------------------------
# Fraud Pattern 4: High-Velocity Transfer Chain
# -----------------------------

def velocity_chain_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] High-Velocity Transfer Chain")

    chain_accounts = accounts_df.sample(5)
    chain_ids = list(chain_accounts["account_id"])

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=11)

    initial_amount = random.randint(50000, 80000)

    # Ensure first account has enough balance
    first_balance = accounts_df.loc[
        accounts_df["account_id"] == chain_ids[0],
        "balance"
    ].values[0]

    if first_balance < initial_amount:
        accounts_df.loc[
            accounts_df["account_id"] == chain_ids[0],
            "balance"
        ] += initial_amount

    current_amount = initial_amount

    for i in range(len(chain_ids) - 1):

        sender = chain_ids[i]
        receiver = chain_ids[i + 1]

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": sender,
            "receiver": receiver,
            "amount": current_amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": accounts_df.loc[
                accounts_df["account_id"] == sender,
                "device_id"
            ].values[0],
            "ip_address": accounts_df.loc[
                accounts_df["account_id"] == sender,
                "ip_address"
            ].values[0]
        }

        attack_transactions.append(transaction)

        # Update balances
        accounts_df.loc[
            accounts_df["account_id"] == sender,
            "balance"
        ] -= current_amount

        accounts_df.loc[
            accounts_df["account_id"] == receiver,
            "balance"
        ] += current_amount

        # Forward 90%
        current_amount = int(current_amount * 0.9)

        transaction_id_start += 1

    # Mark all chain accounts as fraud
    accounts_df.loc[
        accounts_df["account_id"].isin(chain_ids),
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)

    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "High-Velocity Transfer Chain", base_time

# -----------------------------
# Fraud Pattern 5: Cross-Channel Burst
# -----------------------------

def cross_channel_burst_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Cross-Channel Burst Behavior")

    mule_account = accounts_df.sample(1).iloc[0]
    mule_id = mule_account["account_id"]

    recipients = accounts_df.sample(4)

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=12)

    amount = random.randint(20000, 40000)

    for i, recipient in recipients.iterrows():

        if recipient["account_id"] == mule_id:
            continue

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": mule_id,
            "receiver": recipient["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": CHANNELS[i % len(CHANNELS)],
            "device_id": mule_account["device_id"],
            "ip_address": mule_account["ip_address"]
        }

        attack_transactions.append(transaction)

        accounts_df.loc[
            accounts_df["account_id"] == mule_id,
            "balance"
        ] -= amount

        accounts_df.loc[
            accounts_df["account_id"] == recipient["account_id"],
            "balance"
        ] += amount

        transaction_id_start += 1

    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Cross-Channel Burst Behavior", base_time


# -----------------------------
# Fraud Pattern 6: Shared Device Cluster
# -----------------------------

def shared_device_cluster_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Shared Device Cluster")

    cluster_accounts = accounts_df.sample(4)
    cluster_ids = list(cluster_accounts["account_id"])

    shared_device = "D9999"

    accounts_df.loc[
        accounts_df["account_id"].isin(cluster_ids),
        "device_id"
    ] = shared_device

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=13)

    for i in range(len(cluster_ids)):

        sender = cluster_ids[i]
        receiver = cluster_ids[(i + 1) % len(cluster_ids)]

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": sender,
            "receiver": receiver,
            "amount": random.randint(15000, 30000),
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": shared_device,
            "ip_address": accounts_df.loc[
                accounts_df["account_id"] == sender,
                "ip_address"
            ].values[0]
        }

        attack_transactions.append(transaction)
        transaction_id_start += 1

    accounts_df.loc[
        accounts_df["account_id"].isin(cluster_ids),
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Shared Device Cluster", base_time

# -----------------------------
# Fraud Pattern 7: Sudden Behavioral Drift
# -----------------------------

def behavioral_drift_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Sudden Behavioral Drift")

    account = accounts_df.sample(1).iloc[0]
    acc_id = account["account_id"]

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1

    # Simulate sudden burst on day 14
    base_time = START_DATE + timedelta(days=14)

    for i in range(8):  # 8 rapid transactions

        receiver = accounts_df.sample(1).iloc[0]

        if receiver["account_id"] == acc_id:
            continue

        amount = random.randint(30000, 60000)

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": acc_id,
            "receiver": receiver["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": account["device_id"],
            "ip_address": account["ip_address"]
        }

        attack_transactions.append(transaction)
        transaction_id_start += 1

    accounts_df.loc[
        accounts_df["account_id"] == acc_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Sudden Behavioral Drift", base_time


# -----------------------------
# Fraud Pattern 8: Early-Stage Volume Spike
# -----------------------------

def early_volume_spike_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Early-Stage Volume Spike")

    # Create brand new account
    new_account_id = f"A_NEW_{random.randint(1000,9999)}"

    new_account = {
        "account_id": new_account_id,
        "creation_time": START_DATE + timedelta(days=15),
        "device_id": f"D{random.randint(1, 300)}",
        "ip_address": f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
        "balance": 100000,
        "is_fraud": 1
    }

    accounts_df = pd.concat(
        [accounts_df, pd.DataFrame([new_account])],
        ignore_index=True
    )

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=15)

    for i in range(6):

        receiver = accounts_df.sample(1).iloc[0]

        if receiver["account_id"] == new_account_id:
            continue

        amount = random.randint(20000, 40000)

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": new_account_id,
            "receiver": receiver["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": new_account["device_id"],
            "ip_address": new_account["ip_address"]
        }

        attack_transactions.append(transaction)
        transaction_id_start += 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Early-Stage Volume Spike", base_time

# -----------------------------
# Fraud Pattern 9: Smurfing (Structuring)
# -----------------------------

def smurfing_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Smurfing Pattern")

    mule_account = accounts_df.sample(1).iloc[0]
    mule_id = mule_account["account_id"]

    receiver = accounts_df.sample(1).iloc[0]

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1
    base_time = START_DATE + timedelta(days=16)

    # 15 small rapid transactions
    for i in range(15):

        amount = random.randint(2000, 5000)

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": mule_id,
            "receiver": receiver["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": mule_account["device_id"],
            "ip_address": mule_account["ip_address"]
        }

        attack_transactions.append(transaction)
        transaction_id_start += 1

    accounts_df.loc[
        accounts_df["account_id"] == mule_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Smurfing Pattern", base_time

# -----------------------------
# Fraud Pattern 10: Dormant Account Activation
# -----------------------------

def dormant_activation_attack(accounts_df, transactions_df):

    print("\n[Attack Injected] Dormant Account Activation")

    account = accounts_df.sample(1).iloc[0]
    acc_id = account["account_id"]

    attack_transactions = []
    transaction_id_start = len(transactions_df) + 1

    # Simulate inactivity gap (Day 20)
    base_time = START_DATE + timedelta(days=20)

    for i in range(6):

        receiver = accounts_df.sample(1).iloc[0]

        if receiver["account_id"] == acc_id:
            continue

        amount = random.randint(40000, 70000)

        timestamp = base_time + timedelta(minutes=i)

        transaction = {
            "transaction_id": f"T{transaction_id_start}",
            "sender": acc_id,
            "receiver": receiver["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": random.choice(CHANNELS),
            "device_id": account["device_id"],
            "ip_address": account["ip_address"]
        }

        attack_transactions.append(transaction)
        transaction_id_start += 1

    accounts_df.loc[
        accounts_df["account_id"] == acc_id,
        "is_fraud"
    ] = 1

    attack_df = pd.DataFrame(attack_transactions)
    transactions_df = pd.concat([transactions_df, attack_df], ignore_index=True)

    return accounts_df, transactions_df, "Dormant Account Activation", base_time

# attacks.py

attack_registry = [
    fan_in_attack,
    fan_out_attack,
    circular_ring_attack,
    velocity_chain_attack,
    cross_channel_burst_attack,
    shared_device_cluster_attack,
    behavioral_drift_attack,
    early_volume_spike_attack,
    smurfing_attack,
    dormant_activation_attack
]
