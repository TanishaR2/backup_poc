"""CLI entrypoint for the InSightDocs document extraction and ingestion pipeline."""
from pathlib import Path

from app.ingestion.extractor_pipeline import run_extractor_pipeline
from app.ingestion.ingestor_pipeline import run_ingestor_pipeline
from app.generation.generation_service import qna

from utils.logger_config import logger
from utils.settings import COLLECTION_NAME

project_root = Path(__file__).resolve().parent

input_dir = project_root / "data" / "pdf"
output_dir = project_root / "data" / "output"


def run_extraction(
    input_dir: str = str(input_dir),
    output_dir: str = str(output_dir),
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> dict:
    return run_extractor_pipeline(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        collection_name=collection_name,
        force=force,
    )


def run_ingestion(
    output_dir: str = str(output_dir),
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> dict:
    return run_ingestor_pipeline(
        output_dir=Path(output_dir),
        collection_name=collection_name,
        force=force,
    )


def run_rag_pipeline(query: str) -> dict:
    result = qna(query=query)
    logger.info({"Query": result.get("query"), "Response": result.get("answer")})
    return result


if __name__ == "__main__":
    logger.info("Starting document extraction...")
    run_extraction(str(input_dir), str(output_dir))
    logger.success("Extraction completed successfully.")

    logger.info("Starting vector ingestion into Qdrant...")
    run_ingestion(str(output_dir), force=False)
    logger.success("Ingestion completed successfully.")