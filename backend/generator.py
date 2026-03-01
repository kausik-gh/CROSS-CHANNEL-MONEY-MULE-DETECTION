import pandas as pd
import random
from datetime import datetime, timedelta
from backend.config import NUM_ACCOUNTS, NUM_TRANSACTIONS

# Define simulation start date
START_DATE = datetime(2024, 1, 1)

CHANNELS = ["UPI", "App", "ATM", "Web"]

def generate_accounts(num_accounts):
    accounts = []

    for i in range(num_accounts):
        account_id = f"A{str(i+1).zfill(4)}"

        creation_time = START_DATE + timedelta(days=random.randint(0, 365))
        device_id = f"D{random.randint(1, 300)}"
        ip_address = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        balance = random.randint(10000, 200000)

        accounts.append({
            "account_id": account_id,
            "creation_time": creation_time,
            "device_id": device_id,
            "ip_address": ip_address,
            "balance": balance,
            "is_fraud": 0
        })

    return pd.DataFrame(accounts)


def generate_normal_transactions(accounts_df, num_transactions):
    transactions = []
    transaction_id_counter = 1

    for _ in range(num_transactions):

        sender = accounts_df.sample(1).iloc[0]
        receiver = accounts_df.sample(1).iloc[0]

        if sender["account_id"] == receiver["account_id"]:
            continue

        amount = random.randint(500, 10000)

        if sender["balance"] < amount:
            continue

        timestamp = START_DATE + timedelta(
            days=random.randint(0, 7),
            minutes=random.randint(0, 1440)
        )

        channel = random.choice(CHANNELS)

        transactions.append({
            "transaction_id": f"T{transaction_id_counter}",
            "sender": sender["account_id"],
            "receiver": receiver["account_id"],
            "amount": amount,
            "timestamp": timestamp,
            "channel": channel,
            "device_id": sender["device_id"],
            "ip_address": sender["ip_address"]
        })

        accounts_df.loc[
            accounts_df["account_id"] == sender["account_id"], "balance"
        ] -= amount

        accounts_df.loc[
            accounts_df["account_id"] == receiver["account_id"], "balance"
        ] += amount

        transaction_id_counter += 1

    return pd.DataFrame(transactions)
