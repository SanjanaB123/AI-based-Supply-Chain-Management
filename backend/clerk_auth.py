import os
import logging
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import ClerkUser

log = logging.getLogger(__name__)

# Set DEV_BYPASS=true in .env to skip Clerk verification during local development.
# Never enable this in production.
_DEV_BYPASS = os.getenv("DEV_BYPASS", "false").lower() == "true"

# Lazily initialised — created on first request so that load_dotenv() in
# main.py has already run before we read the env var.
_jwks_client: Optional[PyJWKClient] = None

security = HTTPBearer(auto_error=not _DEV_BYPASS)


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        url = os.getenv("CLERK_JWKS_URL")
        if not url:
            raise RuntimeError(
                "CLERK_JWKS_URL environment variable is required. "
                "Set it to https://<your-clerk-instance>.clerk.accounts.dev/.well-known/jwks.json"
            )
        _jwks_client = PyJWKClient(url, cache_keys=True, lifespan=3600)
    return _jwks_client


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> ClerkUser:
    """
    Verify a Clerk-issued JWT from the Authorization header.
    Returns a ClerkUser with the user's Clerk ID and email.
    """
    if _DEV_BYPASS:
        log.warning("DEV_BYPASS enabled — skipping Clerk token verification")
        return ClerkUser(user_id="dev-user", email=os.getenv("SUPPORT_EMAIL", "dev@local.test"))

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    token = credentials.credentials

    try:
        jwks_client = _get_jwks_client()
        log.info(f"Fetching signing key from JWKS for token verification")
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        log.warning(f"Invalid Clerk token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"JWKS/Auth error (type={type(e).__name__}): {e}")
        log.error(f"CLERK_JWKS_URL is: {os.getenv('CLERK_JWKS_URL', 'NOT SET')}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication service error: {type(e).__name__}",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    email = payload.get("email") or payload.get("email_address")

    return ClerkUser(user_id=user_id, email=email)
