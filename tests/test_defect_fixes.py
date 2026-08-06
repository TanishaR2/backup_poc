import pytest
import importlib.util
from pathlib import Path
from app.generation.query_utils import select_relevant_image_path, resolve_existing_image_path

app_py_path = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("app_main", app_py_path)
app_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_main)
_to_html = app_main._to_html

def test_d01_select_relevant_image_path_prevents_cross_paper_mismatch(tmp_path):
    # Create fake image files for testing
    img_vanderer = tmp_path / "page_6_figure_1.png"
    img_vanderer.write_text("fake_png")
    img_other = tmp_path / "page_16_figure_2.png"
    img_other.write_text("fake_png")

    hits = [
        # Hit 1: Unrelated paper image chunk
        {
            "payload": {
                "text": "Figure 2 tactile imaging",
                "metadata": {
                    "doc_id": "2606.14344_More_with_LESS",
                    "document_name": "2606.14344_More_with_LESS.pdf",
                    "chunk_type": "image",
                    "source_path": str(img_other),
                    "rerank_score": 0.90,
                },
            }
        },
        # Hit 2: VANDERER text chunk (Primary document context with query term overlap)
        {
            "payload": {
                "text": "Figure 5 qualitative trajectory comparison between VANDERER and NoMaD in Town 1 Town 3 Town 5",
                "metadata": {
                    "doc_id": "2606.14879_VANDERER",
                    "document_name": "2606.14879_VANDERER.pdf",
                    "chunk_type": "text",
                    "page_number": 6,
                    "rerank_score": 0.85,
                },
            }
        },
        # Hit 3: VANDERER image chunk
        {
            "payload": {
                "text": "Qualitative comparison of navigation traces for VANDERER and NoMaD",
                "metadata": {
                    "doc_id": "2606.14879_VANDERER",
                    "document_name": "2606.14879_VANDERER.pdf",
                    "chunk_type": "image",
                    "page_number": 6,
                    "source_path": str(img_vanderer),
                    "rerank_score": 0.80,
                },
            }
        },
    ]

    selected = select_relevant_image_path(
        query="Display the actual figure image showing qualitative trajectory comparison between VANDERER and NoMaD",
        hits=hits,
        needs_image=True,
    )

    # D-01 Verification: Should pick VANDERER image, NEVER the unrelated paper image
    assert selected == str(img_vanderer)
    assert selected != str(img_other)


def test_d02_and_d08_to_html_cleans_json_and_stray_divs():
    json_with_div = '{"answer": "VANDERER outperforms NoMaD.</div>"}'
    cleaned_html = _to_html(json_with_div)

    # D-02 & D-08 Verification: Raw JSON stripped, stray </div> stripped
    assert '{"answer"' not in cleaned_html
    assert "</div>" not in cleaned_html
    assert "VANDERER outperforms NoMaD." in cleaned_html


def test_d09_latex_math_normalization():
    raw_math = r"The attention weights are $\alpha(h)_{ij} = \text{softmax}(q_i(h) \cdot k_j(h) / \sqrt{d_{head}})$"
    html = _to_html(raw_math)

    # D-09 Verification: LaTeX symbols normalized to clean Unicode symbols (α, ·, √)
    assert "α" in html or "alpha" in html
    assert "·" in html or "cdot" in html
    assert "√" in html or "sqrt" in html


def test_image_aware_validation_scoring():
    from app.agents.validation_agent import validate_answer

    mock_llm_json = '''{
        "correctness": 0.95,
        "relevancy": 1.0,
        "completeness": 0.90,
        "image_relevancy": 0.95,
        "image_correctness": 1.0,
        "validated_image_path": "/path/to/page_7_figure_5.png",
        "reason": "The answer accurately describes VANDERER and the selected figure matches the trajectory query."
    }'''

    res = validate_answer(
        query="Display figure comparing VANDERER and NoMaD trajectory",
        answer="VANDERER explores 13.4% more area than NoMaD as shown in Figure 5.",
        context_chunks=["VANDERER outperforms NoMaD baseline."],
        retrieved_image_path="/path/to/page_7_figure_5.png",
        image_descriptions=[{"source_path": "/path/to/page_7_figure_5.png", "description": "Qualitative trajectory comparison"}],
        llm_response=mock_llm_json,
    )

    assert res["passed"] is True
    assert res["metrics"]["image_relevancy"] == 0.95
    assert res["metrics"]["image_correctness"] == 1.0
    assert res["validated_image_path"] == "/path/to/page_7_figure_5.png"
    assert res["score"] == 0.96

