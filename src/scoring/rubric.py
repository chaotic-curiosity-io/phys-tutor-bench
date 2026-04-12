"""Rubric definitions as structured data for the 6 scoring dimensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreLevel:
    """A single score level within a dimension."""
    score: int
    label: str
    description: str


@dataclass(frozen=True)
class RubricDimension:
    """A single scoring dimension with its levels."""
    id: str
    name: str
    abbreviation: str
    description: str
    levels: tuple[ScoreLevel, ...]

    def get_level(self, score: int) -> ScoreLevel:
        for level in self.levels:
            if level.score == score:
                return level
        raise ValueError(f"Invalid score {score} for dimension {self.id}")


MISCONCEPTION_DIAGNOSIS = RubricDimension(
    id="misconception_diagnosis",
    name="Misconception Diagnosis",
    abbreviation="MD",
    description="How well does the tutor identify and address the student's specific misconception?",
    levels=(
        ScoreLevel(0, "No identification", "Tutor does not identify any misconception; treats the student as simply 'wrong'"),
        ScoreLevel(1, "Misdiagnosis", "Tutor identifies the student is wrong but misdiagnoses the specific misconception"),
        ScoreLevel(2, "Implicit address", "Tutor implicitly addresses the correct misconception but doesn't explicitly surface it"),
        ScoreLevel(3, "Correct identification", "Tutor correctly identifies the misconception category and addresses it directly"),
        ScoreLevel(4, "Expert diagnosis", "Tutor identifies the specific misconception, names it or describes it precisely, and explains why it's a common/natural way to think (validating the student's reasoning process while correcting the conclusion)"),
    ),
)

SCAFFOLDING_STRATEGY = RubricDimension(
    id="scaffolding_strategy",
    name="Scaffolding Strategy",
    abbreviation="SS",
    description="How well does the tutor scaffold the student's learning?",
    levels=(
        ScoreLevel(0, "Lecture/dump", "Tutor lectures / dumps information with no interactive structure"),
        ScoreLevel(1, "Pseudo-Socratic", "Tutor asks questions but they're leading/rhetorical (pseudo-Socratic)"),
        ScoreLevel(2, "Mismatched scaffolding", "Tutor uses genuine scaffolding but mismatched to the student's ZPD (too advanced or too basic)"),
        ScoreLevel(3, "Appropriate scaffolding", "Tutor uses appropriate scaffolding strategy well-matched to the student's current understanding"),
        ScoreLevel(4, "Adaptive bridging", "Tutor uses bridging from anchoring conceptions, calibrates to the student's ZPD, and adjusts strategy based on student responses (adaptive scaffolding)"),
    ),
)

ANSWER_DISCLOSURE_RESTRAINT = RubricDimension(
    id="answer_disclosure_restraint",
    name="Answer Disclosure Restraint",
    abbreviation="ADR",
    description="How well does the tutor avoid simply giving away the answer?",
    levels=(
        ScoreLevel(0, "Immediate disclosure", "Tutor immediately gives the correct answer and explanation"),
        ScoreLevel(1, "Thinly veiled answer", "Tutor gives hints that are effectively the answer with minimal disguise"),
        ScoreLevel(2, "Mostly guides", "Tutor mostly guides but slips into telling at key moments"),
        ScoreLevel(3, "Consistent guidance", "Tutor consistently guides without disclosing, student constructs understanding"),
        ScoreLevel(4, "Productive struggle", "Tutor maintains productive struggle — student arrives at the answer through their own reasoning with minimal nudging"),
    ),
)

CONCEPTUAL_MODEL_BUILDING = RubricDimension(
    id="conceptual_model_building",
    name="Conceptual Model Building",
    abbreviation="CMB",
    description="How well does the tutor help the student build a transferable mental model?",
    levels=(
        ScoreLevel(0, "Answer-focused", "Tutor focuses only on getting the right numerical/factual answer"),
        ScoreLevel(1, "Isolated concept", "Tutor explains the concept but in isolated/memorizable form ('Newton's Third Law says...')"),
        ScoreLevel(2, "Problem-specific", "Tutor connects the concept to the specific problem but doesn't generalize"),
        ScoreLevel(3, "Generalizable model", "Tutor helps student build a generalizable mental model that extends beyond the specific problem"),
        ScoreLevel(4, "Connected transferable model", "Tutor explicitly connects the concept to the student's existing correct knowledge, builds a transferable model, and helps the student see when/where it applies and when it doesn't (boundary conditions)"),
    ),
)

TRANSFER_SUCCESS = RubricDimension(
    id="transfer_success",
    name="Transfer Success",
    abbreviation="TS",
    description="How well does the student apply the concept to a novel transfer problem?",
    levels=(
        ScoreLevel(0, "Not reached / complete failure", "Transfer problem not reached (conversation hit max turns without progress) OR student fails transfer completely"),
        ScoreLevel(1, "Misconception reapplied", "Student attempts transfer but applies the misconception again"),
        ScoreLevel(2, "Partial success", "Student partially succeeds on transfer — gets the principle but can't fully apply it"),
        ScoreLevel(3, "Success with prompting", "Student succeeds on transfer with minor prompting from the tutor"),
        ScoreLevel(4, "Independent transfer", "Student independently and correctly applies the concept to the novel transfer context"),
    ),
)

PEDAGOGICAL_HARM_AVOIDANCE = RubricDimension(
    id="pedagogical_harm_avoidance",
    name="Pedagogical Harm Avoidance",
    abbreviation="PHA",
    description="Does the tutor avoid pedagogically harmful behaviors?",
    levels=(
        ScoreLevel(0, "Active harm", "Tutor actively reinforces the misconception or introduces new ones or uses incorrect physics"),
        ScoreLevel(1, "Pedagogically harmful", "Tutor uses correct physics but makes pedagogically harmful moves (shaming, overwhelming, creating learned helplessness)"),
        ScoreLevel(2, "Neutral", "Tutor is neutral — correct but not actively harmful or helpful"),
        ScoreLevel(3, "Harm avoidant", "Tutor avoids harm and maintains student engagement/motivation"),
        ScoreLevel(4, "Confidence building", "Tutor actively builds student confidence, validates productive aspects of their reasoning, and maintains a safe learning environment throughout"),
    ),
)

# All dimensions in order
ALL_DIMENSIONS = (
    MISCONCEPTION_DIAGNOSIS,
    SCAFFOLDING_STRATEGY,
    ANSWER_DISCLOSURE_RESTRAINT,
    CONCEPTUAL_MODEL_BUILDING,
    TRANSFER_SUCCESS,
    PEDAGOGICAL_HARM_AVOIDANCE,
)

DIMENSION_BY_ID = {d.id: d for d in ALL_DIMENSIONS}
DIMENSION_BY_ABBREV = {d.abbreviation: d for d in ALL_DIMENSIONS}

# Default scoring weights
DEFAULT_WEIGHTS = {
    "misconception_diagnosis": 0.20,
    "scaffolding_strategy": 0.20,
    "answer_disclosure_restraint": 0.15,
    "conceptual_model_building": 0.20,
    "transfer_success": 0.15,
    "pedagogical_harm_avoidance": 0.10,
}


def compute_composite_score(
    scores: dict[str, int],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the weighted composite score from dimension scores.

    Args:
        scores: Map of dimension_id -> score (0-4)
        weights: Optional custom weights (must sum to 1.0)

    Returns:
        Weighted average score (0-4 scale)
    """
    weights = weights or DEFAULT_WEIGHTS
    total = 0.0
    for dim_id, weight in weights.items():
        total += scores.get(dim_id, 0) * weight
    return total


def format_rubric_for_prompt() -> str:
    """Format all rubric dimensions as text for inclusion in LLM judge prompts."""
    lines = []
    for dim in ALL_DIMENSIONS:
        lines.append(f"### {dim.name} ({dim.abbreviation})")
        lines.append(f"{dim.description}\n")
        for level in dim.levels:
            lines.append(f"- **{level.score}** ({level.label}): {level.description}")
        lines.append("")
    return "\n".join(lines)
