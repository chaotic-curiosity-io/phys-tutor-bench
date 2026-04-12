"""Streamlit-based annotation interface for human validation of conversation scores."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from src.scoring.rubric import ALL_DIMENSIONS


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize the SQLite database for storing annotations."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            dimension TEXT NOT NULL,
            score INTEGER NOT NULL CHECK(score >= 0 AND score <= 4),
            justification TEXT,
            timestamp TEXT NOT NULL,
            UNIQUE(annotator_id, conversation_id, dimension)
        )
    """)
    conn.commit()
    return conn


def save_annotation(
    conn: sqlite3.Connection,
    annotator_id: str,
    conversation_id: str,
    scenario_id: str,
    dimension: str,
    score: int,
    justification: str,
) -> None:
    """Save or update an annotation in the database."""
    conn.execute("""
        INSERT INTO annotations (annotator_id, conversation_id, scenario_id, dimension, score, justification, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(annotator_id, conversation_id, dimension)
        DO UPDATE SET score=excluded.score, justification=excluded.justification, timestamp=excluded.timestamp
    """, (
        annotator_id,
        conversation_id,
        scenario_id,
        dimension,
        score,
        justification,
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()


def get_annotations(conn: sqlite3.Connection, conversation_id: str | None = None) -> list[dict]:
    """Retrieve annotations, optionally filtered by conversation_id."""
    query = "SELECT * FROM annotations"
    params: tuple = ()
    if conversation_id:
        query += " WHERE conversation_id = ?"
        params = (conversation_id,)
    query += " ORDER BY timestamp DESC"

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def run_annotation_interface(conversations_dir: str, db_path: str = "data/validation/human_annotations.db") -> None:
    """Launch the Streamlit annotation interface.

    Run with: streamlit run -m src.validation.human_annotation_interface -- --conversations_dir <path>
    """
    st.set_page_config(page_title="PhysTutorBench Annotator", layout="wide")
    st.title("PhysTutorBench — Human Annotation Interface")

    # Load conversations
    conv_dir = Path(conversations_dir)
    conv_files = sorted(conv_dir.rglob("*.json"))

    if not conv_files:
        st.error(f"No conversation files found in {conversations_dir}")
        return

    conn = init_db(db_path)

    # Sidebar: annotator setup
    st.sidebar.header("Annotator Setup")
    annotator_id = st.sidebar.text_input("Your Annotator ID", value="annotator_1")

    # Select conversation
    conv_names = [f.stem for f in conv_files]
    selected_idx = st.sidebar.selectbox("Select Conversation", range(len(conv_names)),
                                         format_func=lambda i: conv_names[i])

    conv_path = conv_files[selected_idx]
    with open(conv_path) as f:
        conv_data = json.load(f)

    # Load scenario if available
    scenario_id = conv_data.get("scenario_id", "unknown")

    # Display scenario context
    st.header("Scenario Context")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Scenario ID:** {scenario_id}")
        st.markdown(f"**Model Under Test:** {conv_data.get('model_under_test', 'unknown')}")
        st.markdown(f"**Total Turns:** {conv_data.get('total_turns', 'N/A')}")
        st.markdown(f"**Termination:** {conv_data.get('termination_reason', 'N/A')}")

    # Display conversation transcript
    st.header("Conversation Transcript")
    for msg in conv_data.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        turn = msg.get("turn_number", "?")

        if role == "tutor":
            st.markdown(f"**[Turn {turn} — TUTOR]**")
            st.info(content)
        else:
            st.markdown(f"**[Turn {turn} — STUDENT]**")
            st.warning(content)

    # Scoring interface
    st.header("Score This Conversation")
    st.markdown("Rate the **tutor's** performance on each dimension (0-4).")

    # Check for existing annotations
    existing = get_annotations(conn, conv_data.get("id"))
    existing_map: dict[str, dict] = {}
    for ann in existing:
        if ann["annotator_id"] == annotator_id:
            existing_map[ann["dimension"]] = ann

    scores_to_save: dict[str, tuple[int, str]] = {}

    for dim in ALL_DIMENSIONS:
        st.subheader(f"{dim.name} ({dim.abbreviation})")

        # Show rubric levels
        with st.expander("View rubric levels"):
            for level in dim.levels:
                st.markdown(f"- **{level.score}** ({level.label}): {level.description}")

        existing_score = existing_map.get(dim.id, {}).get("score", 2)
        existing_just = existing_map.get(dim.id, {}).get("justification", "")

        score = st.slider(
            f"{dim.abbreviation} Score",
            min_value=0,
            max_value=4,
            value=existing_score,
            key=f"score_{dim.id}",
        )
        justification = st.text_area(
            f"{dim.abbreviation} Justification (optional)",
            value=existing_just,
            key=f"just_{dim.id}",
        )
        scores_to_save[dim.id] = (score, justification)

    # Submit
    if st.button("Submit Annotations", type="primary"):
        for dim_id, (score, justification) in scores_to_save.items():
            save_annotation(
                conn,
                annotator_id=annotator_id,
                conversation_id=conv_data.get("id", conv_path.stem),
                scenario_id=scenario_id,
                dimension=dim_id,
                score=score,
                justification=justification,
            )
        st.success("Annotations saved!")

    # Show annotation progress
    st.sidebar.header("Progress")
    all_annotations = get_annotations(conn)
    annotated_convs = set()
    for ann in all_annotations:
        if ann["annotator_id"] == annotator_id:
            annotated_convs.add(ann["conversation_id"])
    st.sidebar.metric("Conversations Annotated", f"{len(annotated_convs)}/{len(conv_files)}")


# Allow running as: python -m src.validation.human_annotation_interface
if __name__ == "__main__":
    import sys
    conversations_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    run_annotation_interface(conversations_dir)
