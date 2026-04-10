import os
from dotenv import load_dotenv

# Load .env from root directory BEFORE any other imports that read env vars at module level
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routes_inventory import router as inventory_router
from routes_chat import router as chat_router, init_ai

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
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(inventory_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    """Top-level health check for Cloud Run / load balancers."""
    from routes_chat import graph
    return {
        "status": "ok",
        "ai_enabled": graph is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
