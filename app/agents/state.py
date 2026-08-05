from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str
    route: str
    retrieved_docs: list
    answer: str
    validation: dict
    final_answer: str

    retrieval_confidence: float
    image_base64: Optional[str]
    chat_history: list
    session_id: str
    needs_web_search: Optional[bool]
    web_snippets: Optional[list]
    is_atomic: Optional[bool]
    needs_image_in_answer: Optional[bool]
    image_description: Optional[str]
    answer_length: Optional[str]  # 'short' | 'medium' | 'detailed'
    scope: Optional[str]  # 'documents' | 'faq'
    retrieved_image_path: Optional[str]