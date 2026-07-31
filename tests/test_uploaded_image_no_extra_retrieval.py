import unittest
from unittest.mock import patch
from app.agents.web_search_tool import fetch_web_image_for_query
from app.generation.query_utils import select_relevant_image_path, generate_support_diagram_image
from api.services import run_query


class TestUploadedImageNoExtraRetrieval(unittest.TestCase):
    def test_empty_query_returns_none_for_all_image_helpers(self):
        self.assertIsNone(fetch_web_image_for_query(""))
        self.assertIsNone(fetch_web_image_for_query("   "))
        self.assertIsNone(select_relevant_image_path("", []))
        self.assertIsNone(generate_support_diagram_image(""))

    @patch("app.agents.nodes.route_query")
    @patch("app.agents.nodes.answer_support_query")
    def test_run_query_with_uploaded_image_returns_no_extra_retrieved_image(self, mock_support, mock_route):
        mock_route.return_value = {
            "route": "support",
            "is_atomic": True,
            "rewritten_query": "Describe and analyze this uploaded image in detail.",
            "answer_length": "detailed",
            "needs_image": False,
        }
        mock_support.return_value = {
            "answer": "This is an ROC curve.",
            "source": "llm_knowledge",
            "retrieved_image_path": None,
        }

        # Query with image attached and empty query text
        res = run_query("", image_base64="fake_base64_data", session_id="test_img_session")
        self.assertIsNone(res.get("retrieved_image_path"))


if __name__ == "__main__":
    unittest.main()
