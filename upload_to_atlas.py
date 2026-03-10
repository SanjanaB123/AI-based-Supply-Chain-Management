from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="Upload CSV to MongoDB Atlas")
parser.add_argument("--file", required=True, help="Path to the CSV file")
args = parser.parse_args()

# Connect to Atlas
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)

# Upload CSV
df = pd.read_csv(args.file)
df = df.where(pd.notnull(df), None)
documents = df.to_dict(orient="records")

collection = client["supply_chain"]["retail_store_inventory"]
collection.insert_many(documents)

print(f"✅ Inserted {len(documents)} documents to Atlas!")