# from pymongo import MongoClient
# from dotenv import load_dotenv
# import os

# load_dotenv()

# client = MongoClient(os.getenv("MONGO_URI"))

# # Test if connected
# print(client.list_database_names())
# print("✅ Connected to MongoDB Atlas!")
import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://vedashreebane_db_user:mlops123@cluster0.w3focfs.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db     = client["supply_chain"]

snap = pd.read_csv("better_inventory_snapshot.csv")
records = snap.to_dict("records")

db["inventory_snapshot"].drop()           # clear if exists
db["inventory_snapshot"].insert_many(records)

print(f"Inserted {len(records)} records into inventory_snapshot")
client.close()