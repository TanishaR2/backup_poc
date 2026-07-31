import base64
from pathlib import Path
from typing import Any

from prompts.prompts import Image_description_prompt
from utils.logger_config import logger
from utils.models_and_clients import openai_client, google_client
from utils.settings import (
    AZURE_OPENAI_MODEL_NAME,
)

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


def _describe_with_azure_openai(image_path: Path, client: Any) -> str:
    """Generate image description using Azure OpenAI."""

    with image_path.open("rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": Image_description_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    }
                ],
            },
        ],
    )

    return response.choices[0].message.content or ""


def _describe_with_gemini(image_path: Path, client: Any) -> str:
    """Generate image description using Gemini."""

    with image_path.open("rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            Image_description_prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
        ],
    )

    return getattr(response, "text", "") or ""


def describe_images(
    image_records: list[dict[str, Any]],
    output_dir: Path,
    provider: Any) -> list[dict[str, Any]]:
    """Generate image descriptions using the selected client."""

    description_dir = output_dir / "image_descriptions"
    description_dir.mkdir(parents=True, exist_ok=True)

    if provider is None:
        logger.warning("Image describer: no provider configured; using placeholder descriptions.")

        for item in image_records:
            item["description"] = (
                "Image description unavailable. No provider is configured."
            )
            item["description_path"] = ""

        return image_records

    for item in image_records:
        try:
            image_path = Path(item["source_path"])
            if provider == "gemini":
                description = _describe_with_gemini(image_path, google_client)

            elif provider in {"openai", "azure_openai"}:
                description = _describe_with_azure_openai(image_path, openai_client)

            else:
                raise ValueError(f"Unsupported client: {provider}")
            

            output_path = description_dir / f"{image_path.stem}.md"
            output_path.write_text(description, encoding="utf-8")

            item["description"] = description
            item["description_path"] = str(output_path)

            logger.success(f"Saved image description -> {output_path}")

        except Exception as exc:
            logger.error(
                f"Image description failed for {item.get('source_path')}: {exc}"
            )
            item["description"] = "Image description unavailable."
            item["description_path"] = ""

    return image_records