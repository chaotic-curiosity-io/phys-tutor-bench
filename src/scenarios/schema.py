"""Pydantic models for scenario validation and data structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TopicArea(str, Enum):
    MECHANICS = "mechanics"
    EM = "em"
    THERMAL_WAVES = "thermal_waves"
    MODERN_QUANTUM = "modern_quantum"


class Level(str, Enum):
    INTRO_ALGEBRA = "intro_algebra"
    INTRO_CALCULUS = "intro_calculus"
    UPPER_DIVISION = "upper_division"


class Affect(str, Enum):
    CONFIDENT_BUT_WRONG = "confident_but_wrong"
    UNCERTAIN_AND_WRONG = "uncertain_and_wrong"
    PARTIALLY_CORRECT = "partially_correct"
    CONFUSED_AND_FRUSTRATED = "confused_and_frustrated"
    DISENGAGED = "disengaged"
    EAGER_BUT_WRONG = "eager_but_wrong"


class ResponseStyle(str, Enum):
    VERBOSE_REASONER = "verbose_reasoner"
    TERSE_ANSWERER = "terse_answerer"
    QUESTION_ASKER = "question_asker"
    FORMULA_PLUGGER = "formula_plugger"
    HEDGING_GUESSER = "hedging_guesser"


class Prevalence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StudentProfile(BaseModel):
    """Describes the simulated student's characteristics."""

    knowledge_level: Level
    knows: list[str] = Field(description="Concepts the student already understands correctly")
    struggles_with: list[str] = Field(description="Specific conceptual difficulties")
    affect: Affect
    response_style: ResponseStyle


class ExpertStrategy(BaseModel):
    """Reference expert tutoring strategy (not shown to models under test)."""

    diagnosis: str = Field(description="Concise description of the misconception and its root")
    recommended_approach: str = Field(description="Pedagogical approach (e.g., bridging_analogy, socratic_questioning)")
    scaffolding_sequence: list[str] = Field(description="Ordered steps an expert tutor would take")
    common_pitfalls: list[str] = Field(description="Mistakes a tutor should avoid")


class Scenario(BaseModel):
    """A single physics tutoring scenario in the benchmark."""

    id: str = Field(description="Unique identifier, e.g. mech-n3l-ar-001")
    topic_area: TopicArea
    subtopic: str
    level: Level

    # The tutoring situation
    problem_context: str = Field(description="The physics problem or situation presented to the student")
    student_initial_response: str = Field(description="The student's first (incorrect) response")

    # Misconception grounding
    misconception_tags: list[str] = Field(description="IDs from the misconception taxonomy")
    misconception_source: str = Field(description="Source instrument (FCI, CSEM, etc.)")
    misconception_description: str = Field(description="Human-readable description of the misconception")
    common_prevalence: Prevalence

    # Student profile
    student_profile: StudentProfile

    # Transfer assessment
    transfer_problem: str = Field(description="A novel problem requiring the same concept in a new context")
    transfer_success_criteria: str = Field(description="What successful transfer looks like")

    # Expert reference
    expert_strategy: ExpertStrategy


class ConversationMessage(BaseModel):
    """A single message in a tutoring conversation."""

    role: str = Field(description="'tutor' or 'student'")
    content: str
    turn_number: int
    timestamp: Optional[str] = None
    token_count: Optional[int] = None


class ConversationRecord(BaseModel):
    """Complete record of a simulated tutoring conversation."""

    id: str = Field(description="Unique conversation ID")
    scenario_id: str
    model_under_test: str = Field(description="Model name/ID of the tutor")
    student_simulator_model: str = Field(description="Model used for student simulation")
    student_simulator_temperature: float

    messages: list[ConversationMessage]
    total_turns: int
    termination_reason: str = Field(description="max_turns, conceptual_change, student_disengaged")

    # Metadata
    tutor_system_prompt: str
    student_system_prompt: str
    run_timestamp: str
    total_tutor_tokens: int = 0
    total_student_tokens: int = 0


class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""

    dimension: str
    score: int = Field(ge=0, le=4)
    justification: str = Field(description="Natural-language explanation referencing specific conversation moments")


class ConversationScore(BaseModel):
    """Complete scoring of a conversation across all dimensions."""

    conversation_id: str
    scenario_id: str
    model_under_test: str
    judge_model: str

    scores: list[DimensionScore]
    composite_score: float = Field(description="Weighted average across dimensions")
    timestamp: str


class MisconceptionEntry(BaseModel):
    """A single entry in the misconception taxonomy."""

    id: str
    description: str
    topic_area: str
    subtopic: str
    source_instrument: Optional[str]
    fci_category: Optional[str] = None
    prevalence: Prevalence
    related_concepts: list[str] = []
    typical_student_reasoning: str = ""
