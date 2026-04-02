import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from database import init_db
from routes_auth import router as auth_router
from routes_inventory import router as inventory_router
from routes_chat import router as chat_router, init_ai

# Load .env from root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

log = logging.getLogger(__name__)

# Initialize MongoDB indexes
try:
    init_db()
    log.info("MongoDB indexes initialized successfully")
except Exception as e:
    log.warning(f"Failed to initialize MongoDB indexes: {e}")


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
app.include_router(auth_router)
app.include_router(inventory_router)
app.include_router(chat_router)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    """Top-level health check for Cloud Run / load balancers."""
    from routes_chat import graph
    return {
        "status": "ok",
        "ai_enabled": graph is not None,
    }


@app.get("/auth-test.html")
def serve_auth_test():
    return FileResponse(os.path.join(STATIC_DIR, "auth-test.html"))
