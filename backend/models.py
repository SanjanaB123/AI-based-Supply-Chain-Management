from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime, timezone
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # nullable for OAuth-only users
    is_active = Column(Boolean, default=True)
    auth_provider = Column(String, default="local")  # "local" or "google"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
