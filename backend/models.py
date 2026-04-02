from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr
import logging

log = logging.getLogger(__name__)

class User(BaseModel):
    """Pydantic model for User data validation and serialization."""
    
    id: Optional[str] = None  # MongoDB ObjectId as string
    email: EmailStr
    username: str
    hashed_password: Optional[str] = None  # nullable for OAuth-only users
    is_active: bool = True
    auth_provider: str = "local"  # "local" or "google"
    created_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}

class UserCreate(BaseModel):
    """Model for user registration requests."""
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    """Model for user login requests."""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """Model for user response data (excluding sensitive fields)."""
    id: str
    email: str
    username: str
    is_active: bool
    auth_provider: str
    created_at: datetime

    model_config = {"from_attributes": True}

class Token(BaseModel):
    """Model for JWT token response."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Model for JWT token data."""
    email: Optional[str] = None

class UserDocument:
    """
    Helper class for MongoDB user document operations.
    Provides static methods for converting between Pydantic models and MongoDB documents.
    """
    
    @staticmethod
    def to_document(user: UserCreate, hashed_password: str, auth_provider: str = "local") -> Dict[str, Any]:
        """Convert UserCreate to MongoDB document."""
        return {
            "email": user.email,
            "username": user.username,
            "hashed_password": hashed_password,
            "is_active": True,
            "auth_provider": auth_provider,
            "created_at": datetime.now(timezone.utc)
        }
    
    @staticmethod
    def from_document(doc: Dict[str, Any]) -> User:
        """Convert MongoDB document to User model."""
        return User(
            id=str(doc["_id"]),
            email=doc["email"],
            username=doc["username"],
            hashed_password=doc.get("hashed_password"),
            is_active=doc.get("is_active", True),
            auth_provider=doc.get("auth_provider", "local"),
            created_at=doc.get("created_at")
        )
    
    @staticmethod
    def to_response(doc: Dict[str, Any]) -> UserResponse:
        """Convert MongoDB document to UserResponse model."""
        return UserResponse(
            id=str(doc["_id"]),
            email=doc["email"],
            username=doc["username"],
            is_active=doc.get("is_active", True),
            auth_provider=doc.get("auth_provider", "local"),
            created_at=doc.get("created_at")
        )
