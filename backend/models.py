from typing import Optional

from pydantic import BaseModel


class ClerkUser(BaseModel):
    """User info extracted from a verified Clerk JWT."""
    user_id: str
    email: Optional[str] = None
