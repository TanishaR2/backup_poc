import pytest
import importlib.util
from pathlib import Path

app_py_path = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("app_ui_module", app_py_path)
app_ui_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_ui_module)
_to_html = app_ui_module._to_html

def test_to_html_formatting():
    html = _to_html("**InSightDocs** is great")
    assert "InSightDocs" in html

def test_to_html_latex_normalization():
    html = _to_html(r"Formula: \alpha \rightarrow \beta")
    assert "α" in html
    assert "→" in html
    assert "β" in html

def test_to_html_empty():
    assert _to_html("") == ""
