from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
from typing import Generator, Dict, Any
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
users_collection = db["users"]

log = logging.getLogger(__name__)

def get_db() -> Generator[Dict[str, Any], None, None]:
    """
    Get MongoDB database instance for user operations.
    
    Yields:
        MongoDB database instance
    """
    try:
        # Test connection
        client.admin.command("ping")
        log.info("Successfully connected to MongoDB for user operations")
        yield db
    except ConnectionFailure as e:
        log.error(f"Failed to connect to MongoDB: {e}")
        raise RuntimeError("Could not connect to MongoDB") from e
    except Exception as e:
        log.error(f"Unexpected error with MongoDB connection: {e}")
        raise

def get_users_collection():
    """
    Get the users collection directly.
    
    Returns:
        MongoDB collection for users
    """
    return users_collection

def init_db():
    """
    Initialize database indexes for users collection.
    Call this during application startup.
    """
    try:
        # Create indexes for better performance and uniqueness
        users_collection.create_index("email", unique=True)
        users_collection.create_index("username", unique=True)
        users_collection.create_index("created_at")
        log.info("Database indexes created successfully")
    except Exception as e:
        log.error(f"Failed to create database indexes: {e}")
        raise

# Test connection on module import
try:
    client.admin.command("ping")
    log.info("MongoDB connection established")
except Exception as e:
    log.warning(f"MongoDB connection failed: {e}")
