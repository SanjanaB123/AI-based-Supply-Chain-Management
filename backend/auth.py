import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt
import logging

from database import get_users_collection
from models import User, UserDocument, TokenData

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required but not set")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Google OAuth settings
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# ── Password hashing ────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ── JWT helpers ──────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        token: JWT token from OAuth2PasswordBearer
        
    Returns:
        User object if authentication successful
        
    Raises:
        HTTPException if authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Get user from MongoDB
    users_collection = get_users_collection()
    user_doc = users_collection.find_one({"email": email})
    
    if user_doc is None:
        raise credentials_exception
    
    return UserDocument.from_document(user_doc)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: User from get_current_user
        
    Returns:
        User if active
        
    Raises:
        HTTPException if user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ── User CRUD Operations ─────────────────────────────────
async def create_user(user_data: dict) -> User:
    """
    Create a new user in MongoDB.
    
    Args:
        user_data: User document data
        
    Returns:
        Created User object
    """
    users_collection = get_users_collection()
    
    try:
        result = users_collection.insert_one(user_data)
        user_doc = users_collection.find_one({"_id": result.inserted_id})
        return UserDocument.from_document(user_doc)
    except Exception as e:
        log.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


async def get_user_by_email(email: str) -> Optional[User]:
    """
    Get a user by email from MongoDB.
    
    Args:
        email: User email
        
    Returns:
        User object if found, None otherwise
    """
    users_collection = get_users_collection()
    user_doc = users_collection.find_one({"email": email})
    
    if user_doc:
        return UserDocument.from_document(user_doc)
    return None


async def get_user_by_username(username: str) -> Optional[User]:
    """
    Get a user by username from MongoDB.
    
    Args:
        username: Username
        
    Returns:
        User object if found, None otherwise
    """
    users_collection = get_users_collection()
    user_doc = users_collection.find_one({"username": username})
    
    if user_doc:
        return UserDocument.from_document(user_doc)
    return None


async def update_user(user_id: str, update_data: dict) -> Optional[User]:
    """
    Update a user in MongoDB.
    
    Args:
        user_id: User ID (MongoDB ObjectId as string)
        update_data: Data to update
        
    Returns:
        Updated User object if successful, None otherwise
    """
    users_collection = get_users_collection()
    
    try:
        from bson.objectid import ObjectId
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)}, 
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
            return UserDocument.from_document(user_doc)
        return None
    except Exception as e:
        log.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")
