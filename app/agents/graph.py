"""Graph assembly and compilation for InSightDocs LangGraph workflow."""

import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from utils.logger_config import logger
from app.agents.state import AgentState
from app.agents.nodes import (
    planner_node,
    retrieval_node,
    generation_node,
    validation_node,
    support_node,
    final_node,
    faq_node,
)
from app.agents.edges import (
    route_after_retrieval,
    should_validate,
    route_after_validation,
)


def create_sqlite_saver(db_path: str = None) -> SqliteSaver:
    """Create a persistent SqliteSaver instance using a persistent connection."""
    if db_path is None:
        checkpoint_dir = Path(__file__).resolve().parents[2] / "data"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(checkpoint_dir / "checkpoints.db")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return SqliteSaver(conn)


def build_graph(checkpointer=None) -> object:
    """Build and compile the InSightDocs agent graph."""
    logger.info("[Graph] Assembling Agent Graph...")

    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("planner", planner_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generation", generation_node)
    g.add_node("validation", validation_node)
    g.add_node("support", support_node)
    g.add_node("final", final_node)
    g.add_node("faq", faq_node)

    # Set entry point
    g.set_entry_point("planner")

    # Conditional edges from planner
    g.add_conditional_edges(
        "planner",
        lambda state: state["route"],
        {
            "rag": "retrieval",
            "support": "support",
            "faq": "retrieval",
        },
    )

    # Conditional edges from retrieval
    g.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "generation": "generation",
            "support": "support",
        },
    )

    # Conditional edges from generation
    g.add_conditional_edges(
        "generation",
        should_validate,
        {
            "validate": "validation",
            "skip": "final",
        },
    )

    # Conditional edges from validation
    g.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "final": "final",
            "support": "support",
        },
    )

    # Terminal edges
    g.add_edge("final", END)
    g.add_edge("support", END)
    g.add_edge("faq", END)


    if checkpointer is None:
        try:
            checkpointer = create_sqlite_saver()
        except Exception as e:
            logger.warning(f"Failed to initialize SqliteSaver ({e}), falling back to MemorySaver")
            checkpointer = MemorySaver()

    app = g.compile(checkpointer=checkpointer)
    logger.success("[Graph] Agent Graph compiled successfully")
    return app


app = build_graph()
