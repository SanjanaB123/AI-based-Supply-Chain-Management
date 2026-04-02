import httpx
import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

# In-memory store for OAuth state tokens (use Redis in production for multi-process)
_oauth_states: set[str] = set()

from models import UserCreate, UserLogin, UserResponse, Token, UserDocument
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_active_user, create_user, get_user_by_email, get_user_by_username,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI,
)

log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Register ─────────────────────────────────────────────
@router.post("/register", response_model=UserResponse, status_code=201)
@limiter.limit("5/minute")
async def register(request: Request, user_in: UserCreate):
    """
    Register a new user with email and password.
    
    Args:
        user_in: User registration data
        
    Returns:
        Created user information
        
    Raises:
        HTTPException if email or username already exists
    """
    # Check if email already exists
    existing_user = await get_user_by_email(user_in.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if username already exists
    existing_username = await get_user_by_username(user_in.username)
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    # Create user document
    hashed_password = hash_password(user_in.password)
    user_doc = UserDocument.to_document(user_in, hashed_password, auth_provider="local")
    
    try:
        user = await create_user(user_doc)
        log.info(f"User registered successfully: {user.email}")
        return UserDocument.to_response(user.model_dump())
    except Exception as e:
        log.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


# ── Login (JWT) ──────────────────────────────────────────
@router.post("/login", response_model=Token)
@limiter.limit("10/minute")
async def login(request: Request, user_in: UserLogin):
    """
    Authenticate user and return JWT token.
    
    Args:
        user_in: User login credentials
        
    Returns:
        JWT access token
        
    Raises:
        HTTPException if credentials are invalid
    """
    user = await get_user_by_email(user_in.email)
    
    if not user or not user.hashed_password or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(data={"sub": user.email})
    log.info(f"User logged in successfully: {user.email}")
    return {"access_token": token, "token_type": "bearer"}


# ── Current user (protected) ────────────────────────────
@router.get("/me", response_model=UserResponse)
async def read_current_user(current_user = Depends(get_current_active_user)):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Authenticated user from JWT token
        
    Returns:
        Current user information
    """
    return UserDocument.to_response(current_user.model_dump())


# ── Google OAuth: redirect to Google ─────────────────────
@router.get("/google/login")
async def google_login():
    """
    Redirect user to Google OAuth login page.
    
    Returns:
        Redirect to Google OAuth URL
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID env var.")

    state = secrets.token_urlsafe(32)
    _oauth_states.add(state)

    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
        "&access_type=offline"
        f"&state={state}"
    )
    return RedirectResponse(url=google_auth_url)


# ── Google OAuth: callback ───────────────────────────────
@router.get("/google/callback")
async def google_callback(code: str, state: str = ""):
    """
    Handle Google OAuth callback and create/update user.

    Args:
        code: Authorization code from Google
        state: CSRF state token for validation

    Returns:
        Redirect to frontend with JWT token
    """
    if not state or state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or missing OAuth state parameter")
    _oauth_states.discard(state)

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code with Google")

        token_data = token_resp.json()
        id_token = token_data.get("id_token")

        # Get user info from Google
        async with httpx.AsyncClient() as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
        
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")

        google_user = userinfo_resp.json()
        email = google_user["email"]
        name = google_user.get("name", email.split("@")[0])

        # Find or create user
        user = await get_user_by_email(email)
        
        if not user:
            # Create new Google OAuth user
            from models import UserCreate
            user_create = UserCreate(
                email=email,
                username=name,
                password=""  # No password for OAuth users
            )
            
            user_doc = UserDocument.to_document(user_create, None, auth_provider="google")
            user = await create_user(user_doc)
            log.info(f"New Google OAuth user created: {email}")
        else:
            log.info(f"Existing Google OAuth user logged in: {email}")

        jwt_token = create_access_token(data={"sub": user.email})

        # Redirect to frontend — set token in HTTP-only cookie instead of URL
        response = RedirectResponse(url="/auth-test.html", status_code=302)
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600,
        )
        return response
        
    except Exception as e:
        log.error(f"Google OAuth callback failed: {e}")
        raise HTTPException(status_code=500, detail="OAuth authentication failed")


# ── Health check ─────────────────────────────────────────
@router.get("/health")
async def auth_health():
    """
    Check authentication system health.
    
    Returns:
        Health status of authentication system
    """
    try:
        # Test MongoDB connection
        from database import get_users_collection
        users_collection = get_users_collection()
        users_collection.count_documents({})
        
        return {
            "status": "healthy",
            "database": "connected",
            "oauth_configured": bool(GOOGLE_CLIENT_ID)
        }
    except Exception as e:
        log.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }
