import pytest
from pathlib import Path
from app.ingestion.document_validator import validate_document_for_ingestion

def test_non_pdf_file_rejected(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("This is a plain text file, not a PDF.")
    
    result = validate_document_for_ingestion(txt_file)
    assert result["valid"] is False
    assert "PDF files" in result["reason"]

def test_file_size_exceeded(tmp_path):
    large_pdf = tmp_path / "large.pdf"
    # Write dummy content larger than 10MB
    with open(large_pdf, "wb") as f:
        f.seek(11 * 1024 * 1024 - 1)
        f.write(b"\0")
        
    result = validate_document_for_ingestion(large_pdf)
    assert result["valid"] is False
    assert "exceeds" in result["reason"].lower()
