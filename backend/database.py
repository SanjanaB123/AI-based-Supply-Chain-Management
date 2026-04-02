from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "inventory_forecasting")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set")

# Initialize MongoDB client
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]

log = logging.getLogger(__name__)

# Test connection on module import
try:
    client.admin.command("ping")
    log.info("MongoDB connection established")
except Exception as e:
    log.warning(f"MongoDB connection failed: {e}")
