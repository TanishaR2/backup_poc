from pathlib import Path
from typing import Any

from prompts.prompts import Table_description_prompt
from utils.logger_config import logger

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None


def describe_tables(table_records: list[dict[str, Any]], output_dir: Path, groq_client: Any | None = None) -> list[dict[str, Any]]:
    """Generate table descriptions using Groq and save them as markdown files."""
    description_dir = output_dir / "table_descriptions"
    description_dir.mkdir(parents=True, exist_ok=True)

    if groq_client is None or Groq is None:
        logger.warning("Groq client is not available; using placeholder table descriptions")
        for item in table_records:
            item["description"] = "Table description unavailable. Groq client is not configured."
            item["description_path"] = ""
        return table_records

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        
    ]

    for item in table_records:
        try:
            table_path = Path(item["source_path"])
            table_md = table_path.read_text(encoding="utf-8")
            response = None

            for model in models:
                try:
                    response = groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": Table_description_prompt.format(table_content=table_md)}],
                        temperature=0,
                    )
                    break
                except Exception as exc:
                    logger.error(f"Table description model {model} failed: {exc}")

            if response is None:
                try:
                    logger.info("Groq failed for table description; falling back to OpenAI")
                    from utils.models_and_clients import openai_client
                    from utils.settings import AZURE_OPENAI_MODEL_NAME
                    if openai_client is not None:
                        response = openai_client.chat.completions.create(
                            model=AZURE_OPENAI_MODEL_NAME,
                            messages=[{"role": "user", "content": Table_description_prompt.format(table_content=table_md)}],
                        )
                except Exception as exc:
                    logger.error(f"Table description OpenAI fallback failed: {exc}")

            description = response.choices[0].message.content if response is not None else "Table description unavailable."

            output_path = description_dir / f"{table_path.stem}.md"
            output_path.write_text(description, encoding="utf-8")

            item["description"] = description
            item["description_path"] = str(output_path)
            logger.success(f"Saved table description -> {output_path}")
        except Exception as exc:
            logger.error(f"Table description failed for {item.get('source_path')}: {exc}")
            item["description"] = "Table description unavailable."
            item["description_path"] = ""

    return table_records
