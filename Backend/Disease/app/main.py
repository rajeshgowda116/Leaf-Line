from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.config import settings
from app.api.routes import router as disease_router

# Path to Frountend templates
FRONTEND_TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "Frountend" / "templates"
)


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

    # Convenience routes to serve the Frontend templates directly
    @application.get("/", tags=["Frontend"], summary="Serve Home Page")
    async def serve_home():
        home_path = FRONTEND_TEMPLATES_DIR / "home.html"
        if home_path.exists():
            return FileResponse(home_path, media_type="text/html")
        return HTMLResponse("<h1>TreeTrack AI Disease Detection Backend is Running.</h1><p>Visit <a href='/docs'>/docs</a> for the API.</p>")

    @application.get("/chat", tags=["Frontend"], summary="Serve Disease Chat Page")
    async def serve_chat():
        chat_path = FRONTEND_TEMPLATES_DIR / "Chat.html.html"
        if chat_path.exists():
            return FileResponse(chat_path, media_type="text/html")
        return HTMLResponse("<h1>Chat Template not found</h1>")

    return application


app = create_application()
