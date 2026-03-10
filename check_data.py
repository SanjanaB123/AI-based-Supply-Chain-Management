from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGO_URI")
print(f"Connecting to: {uri}")

client = MongoClient(uri)
db = client["supply_chain"]
collection = db["retail_store_inventory"]

count = collection.count_documents({})
print(f"Total documents in collection: {count}")