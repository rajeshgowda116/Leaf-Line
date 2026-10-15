import sys
from pathlib import Path

# Ensure Backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from config import settings
from api.routes import router as disease_router


def get_frontend_templates_dir() -> Path:
    """Finds Frountend/templates across parent directories."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        for folder in ("Frountend", "Frontend"):
            candidate = parent / folder / "templates"
            if candidate.exists():
                return candidate
    return current.parent.parent / "Frountend" / "templates"


FRONTEND_TEMPLATES_DIR = get_frontend_templates_dir()


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="TreeTrack AI Plant Disease Detection Service supporting Tomato, Chilli, Brinjal, Okra, and Cucumber.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 router
    application.include_router(disease_router, prefix=settings.API_V1_PREFIX)

    @application.get("/health", tags=["Health"], summary="Service Health Check")
    async def health_check():
        """Returns the health status of the Disease Detection service."""
        return {
            "status": "healthy",
            "service": "disease-detection",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        }

    # Convenience routes to serve Frontend templates directly
    @application.get("/", tags=["Frontend"], summary="Serve Home Page")
    async def serve_home():
        home_path = FRONTEND_TEMPLATES_DIR / "home.html"
        if home_path.exists():
            return FileResponse(home_path, media_type="text/html")
        return HTMLResponse("<h1>TreeTrack AI Backend Running</h1><p><a href='/docs'>Swagger API Docs</a></p>")

    @application.get("/chat", tags=["Frontend"], summary="Serve Disease Chat Page")
    async def serve_chat():
        for filename in ("Chat.html", "Chat.html.html"):
            chat_path = FRONTEND_TEMPLATES_DIR / filename
            if chat_path.exists():
                return FileResponse(chat_path, media_type="text/html")
        return HTMLResponse("<h1>Chat Template not found</h1>")

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
