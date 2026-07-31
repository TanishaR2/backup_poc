from fastapi import FastAPI

from contextlib import asynccontextmanager

from .routes import router
from utils.logger_config import logger

logger.info("Started")
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.success("🚀 InSightDocs API Started")
    logger.info("Swagger Docs: http://127.0.0.1:8000/docs")
    logger.info("=" * 60)
    yield
    logger.warning("🛑 InSightDocs API Stopped")


app = FastAPI(
    title="InSightDocs API",
    description="Multimodal RAG over research PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {
        "name": "InSightDocs API",
        "docs": "/docs",
    }