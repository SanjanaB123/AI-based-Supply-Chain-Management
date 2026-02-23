from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

# Test if connected
print(client.list_database_names())
print("✅ Connected to MongoDB Atlas!")