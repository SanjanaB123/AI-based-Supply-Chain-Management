import os
from dotenv import load_dotenv

# Load .env from root directory BEFORE any other imports that read env vars at module level
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routes_inventory import router as inventory_router
from routes_gemini_chat import router as gemini_chat_router
from routes_email import router as email_router

# routes_chat requires heavy ML deps (setfit, langchain, etc.)
# Gracefully skip if not installed (e.g. local dev without full ML stack)
try:
    from routes_chat import router as chat_router, init_ai
    _chat_available = True
except ImportError as _e:
    logging.getLogger(__name__).warning(f"Claude chat unavailable (missing deps): {_e}")
    chat_router = None  # type: ignore
    _chat_available = False

    async def init_ai():
        pass

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_ai()
    except Exception as e:
        log.error(f"Error initializing AI/ML components: {e}")
        log.error("Chat functionality will be disabled.")
    yield


app = FastAPI(
    title="Inventory Dashboard API",
    version="1.0.0",
    description="Integrated Inventory Dashboard with AI Chat capabilities",
    lifespan=lifespan,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://frontend-prod-pbgmiffqlq-uc.a.run.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(inventory_router)
if chat_router is not None:
    app.include_router(chat_router)
app.include_router(gemini_chat_router)
app.include_router(email_router)


@app.get("/health")
def health():
    """Top-level health check for Cloud Run / load balancers."""
    ai_enabled = False
    if _chat_available:
        from routes_chat import graph
        ai_enabled = graph is not None
    return {
        "status": "ok",
        "ai_enabled": ai_enabled,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
