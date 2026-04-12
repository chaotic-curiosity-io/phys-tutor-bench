"""Discriminant validity — testing the benchmark with known-quality conversations.

Generates "gold standard" and "foil" conversations to verify the rubric
differentiates between them as expected.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import anthropic
import numpy as np
from rich.console import Console
from rich.table import Table

from src.scoring.rubric import ALL_DIMENSIONS
from src.scoring.llm_judge import LLMJudge
from src.scenarios.schema import (
    Scenario,
    ConversationRecord,
    ConversationMessage,
    ConversationScore,
)

console = Console()

# Profiles for generating foil conversations
FOIL_PROFILES = {
    "answer_dumper": {
        "description": "Tutor immediately explains the correct answer in full detail",
        "tutor_system_prompt": (
            "You are explaining physics to a student. When they give a wrong answer, "
            "immediately explain the correct answer in full detail with all the reasoning. "
            "Be thorough and complete. Give the full derivation and explanation right away."
        ),
        "expected_pattern": {
            "misconception_diagnosis": (2, 4),    # Likely identifies it while explaining
            "scaffolding_strategy": (0, 1),        # No scaffolding, just lecturing
            "answer_disclosure_restraint": (0, 1), # Gives away the answer immediately
            "conceptual_model_building": (1, 2),   # May explain concept but not build student's model
            "transfer_success": (0, 2),            # Student may parrot but not truly transfer
            "pedagogical_harm_avoidance": (2, 4),  # Not actively harmful, just unhelpful
        },
    },
    "wrong_physics": {
        "description": "Tutor uses confident but incorrect physics",
        "tutor_system_prompt": (
            "You are a physics tutor, but you have some fundamental misconceptions yourself. "
            "When the student gives a wrong answer, you 'correct' them, but your corrections "
            "are also wrong. You are confident and clear in your (incorrect) explanations. "
            "For Newton's Third Law, you believe the bigger object exerts more force. "
            "For energy, you believe energy can be created or destroyed. "
            "For circuits, you believe current is used up by resistors. "
            "Present your incorrect physics confidently."
        ),
        "expected_pattern": {
            "misconception_diagnosis": (0, 1),     # May identify student is wrong but for wrong reasons
            "scaffolding_strategy": (0, 2),        # May scaffold but toward wrong conclusions
            "answer_disclosure_restraint": (0, 2), # Variable
            "conceptual_model_building": (0, 1),   # Builds wrong model
            "transfer_success": (0, 1),            # Wrong transfer
            "pedagogical_harm_avoidance": (0, 1),  # Actively harmful — incorrect physics
        },
    },
    "pseudo_socratic": {
        "description": "Tutor asks leading rhetorical questions that effectively give away the answer",
        "tutor_system_prompt": (
            "You are a physics tutor who tries to use the Socratic method but does it poorly. "
            "Ask leading questions where the answer is obvious from the question itself. "
            "For example: 'Don't you think that since Newton's Third Law says forces are equal, "
            "the forces must be equal here too?' Your questions should be rhetorical — "
            "they lead the student directly to the answer without requiring genuine thinking."
        ),
        "expected_pattern": {
            "misconception_diagnosis": (1, 3),     # May identify misconception through questions
            "scaffolding_strategy": (1, 2),        # Pseudo-scaffolding
            "answer_disclosure_restraint": (1, 2), # Giving away answer through questions
            "conceptual_model_building": (1, 2),   # Limited model building
            "transfer_success": (1, 3),            # Variable
            "pedagogical_harm_avoidance": (2, 4),  # Not harmful, just ineffective
        },
    },
    "expert_tutor": {
        "description": "Expert tutor using PER best practices",
        "tutor_system_prompt": (
            "You are an expert physics tutor trained in Physics Education Research (PER). "
            "When a student exhibits a misconception:\n"
            "1. First, validate what's CORRECT about their reasoning\n"
            "2. Identify the specific misconception without lecturing\n"
            "3. Use bridging analogies — start from an 'anchoring conception' the student "
            "already holds correctly and build a conceptual bridge to the target concept\n"
            "4. Ask genuine (not rhetorical) questions that require the student to think\n"
            "5. Help the student arrive at the correct understanding themselves\n"
            "6. Build a transferable mental model with explicit boundary conditions\n"
            "7. Maintain student confidence and engagement throughout\n"
            "Never give away the answer. Maintain productive struggle."
        ),
        "expected_pattern": {
            "misconception_diagnosis": (3, 4),
            "scaffolding_strategy": (3, 4),
            "answer_disclosure_restraint": (3, 4),
            "conceptual_model_building": (3, 4),
            "transfer_success": (2, 4),
            "pedagogical_harm_avoidance": (3, 4),
        },
    },
}


def generate_foil_conversation(
    scenario: Scenario,
    foil_type: str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 10,
    client: anthropic.Anthropic | None = None,
) -> ConversationRecord:
    """Generate a foil conversation with a specific tutor behavior pattern.

    Uses a simple alternating generation rather than the full engine,
    since we need precise control over the tutor's behavior.
    """
    client = client or anthropic.Anthropic()
    profile = FOIL_PROFILES[foil_type]
    tutor_prompt = profile["tutor_system_prompt"]

    # Student system prompt (simplified)
    student_prompt = (
        f"You are a physics student with this misconception: {scenario.misconception_description}\n"
        f"You are {scenario.student_profile.affect.value}. "
        f"Respond naturally as a real student would. Hold your misconception unless "
        f"genuinely convinced otherwise through good reasoning."
    )

    messages: list[ConversationMessage] = []
    tutor_history: list[dict[str, str]] = []
    student_history: list[dict[str, str]] = []

    # Initial student message
    messages.append(ConversationMessage(
        role="student",
        content=scenario.student_initial_response,
        turn_number=0,
    ))

    opening = (
        f"Problem: {scenario.problem_context}\n\n"
        f"Student says: {scenario.student_initial_response}"
    )
    tutor_history.append({"role": "user", "content": opening})

    for turn in range(1, max_turns + 1):
        # Tutor
        tutor_resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.7,
            system=tutor_prompt,
            messages=tutor_history,
        )
        tutor_text = tutor_resp.content[0].text
        messages.append(ConversationMessage(role="tutor", content=tutor_text, turn_number=turn))

        tutor_history.append({"role": "assistant", "content": tutor_text})
        student_history.append({"role": "user", "content": tutor_text})

        if turn >= max_turns:
            break

        # Student
        student_resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.7,
            system=student_prompt,
            messages=student_history,
        )
        student_text = student_resp.content[0].text
        messages.append(ConversationMessage(role="student", content=student_text, turn_number=turn))

        student_history.append({"role": "assistant", "content": student_text})
        tutor_history.append({"role": "user", "content": student_text})

    return ConversationRecord(
        id=f"foil-{foil_type}-{str(uuid4())[:8]}",
        scenario_id=scenario.id,
        model_under_test=f"foil_{foil_type}",
        student_simulator_model=model,
        student_simulator_temperature=0.7,
        messages=messages,
        total_turns=max_turns,
        termination_reason="max_turns",
        tutor_system_prompt=tutor_prompt,
        student_system_prompt=student_prompt,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_discriminant_validity(
    scenarios: list[Scenario],
    judge: LLMJudge,
    output_dir: str | Path = "data/validation",
    n_scenarios: int = 5,
    generation_model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run discriminant validity testing.

    Generates gold and foil conversations, scores them, and checks
    that they rank in the expected order.
    """
    output_dir = Path(output_dir)
    test_scenarios = scenarios[:n_scenarios]
    results: dict[str, list[ConversationScore]] = {ft: [] for ft in FOIL_PROFILES}

    console.print(f"\n[bold]Discriminant Validity Testing[/bold]")
    console.print(f"Using {n_scenarios} scenarios, {len(FOIL_PROFILES)} tutor profiles\n")

    for scenario in test_scenarios:
        console.print(f"Scenario: {scenario.id}")
        for foil_type in FOIL_PROFILES:
            console.print(f"  Generating {foil_type} conversation...")
            conv = generate_foil_conversation(
                scenario, foil_type, model=generation_model
            )

            # Save conversation
            conv_dir = output_dir / f"{foil_type}_conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            conv_path = conv_dir / f"{conv.id}.json"
            conv_path.write_text(json.dumps(conv.model_dump(), indent=2))

            # Score it
            console.print(f"  Scoring {foil_type}...")
            score = judge.score(conv, scenario)
            results[foil_type].append(score)

            # Save score
            score_path = conv_dir / f"{conv.id}_score.json"
            score_path.write_text(json.dumps(score.model_dump(), indent=2))

    # Analyze results
    analysis = _analyze_discriminant_results(results)
    return analysis


def _analyze_discriminant_results(
    results: dict[str, list[ConversationScore]],
) -> dict:
    """Analyze whether foil types are differentiated as expected."""
    analysis: dict = {}

    # Compute averages per foil type per dimension
    summary_table = Table(title="Discriminant Validity — Average Scores by Tutor Type")
    summary_table.add_column("Tutor Type")
    for dim in ALL_DIMENSIONS:
        summary_table.add_column(dim.abbreviation, justify="right")
    summary_table.add_column("Composite", justify="right", style="bold")

    type_averages: dict[str, dict[str, float]] = {}

    for foil_type, scores in results.items():
        if not scores:
            continue

        avgs: dict[str, float] = {}
        for dim in ALL_DIMENSIONS:
            values = []
            for cs in scores:
                for ds in cs.scores:
                    if ds.dimension == dim.id:
                        values.append(ds.score)
            avgs[dim.id] = float(np.mean(values)) if values else 0.0

        composite = float(np.mean([s.composite_score for s in scores]))
        avgs["composite"] = composite
        type_averages[foil_type] = avgs

        row = [foil_type]
        for dim in ALL_DIMENSIONS:
            row.append(f"{avgs[dim.id]:.2f}")
        row.append(f"{composite:.2f}")
        summary_table.add_row(*row)

    console.print(summary_table)

    # Check expected ordering: expert > pseudo_socratic > answer_dumper > wrong_physics
    expected_order = ["expert_tutor", "pseudo_socratic", "answer_dumper", "wrong_physics"]
    available_order = [t for t in expected_order if t in type_averages]

    if len(available_order) >= 2:
        composites = [type_averages[t].get("composite", 0) for t in available_order]
        is_ordered = all(composites[i] >= composites[i + 1] for i in range(len(composites) - 1))
        analysis["expected_order_maintained"] = is_ordered
        analysis["composite_order"] = list(zip(available_order, composites))

        if is_ordered:
            console.print("\n[green]Expected ordering maintained: expert > pseudo_socratic > answer_dumper > wrong_physics[/green]")
        else:
            console.print("\n[yellow]Warning: Expected ordering NOT maintained[/yellow]")
            for t, c in zip(available_order, composites):
                console.print(f"  {t}: {c:.2f}")

    # Check per-dimension expected patterns
    violations = []
    for foil_type, profile in FOIL_PROFILES.items():
        if foil_type not in type_averages:
            continue
        for dim_id, (expected_low, expected_high) in profile["expected_pattern"].items():
            actual = type_averages[foil_type].get(dim_id, 0)
            if actual < expected_low - 0.5 or actual > expected_high + 0.5:
                violations.append({
                    "foil_type": foil_type,
                    "dimension": dim_id,
                    "expected_range": f"{expected_low}-{expected_high}",
                    "actual": actual,
                })

    analysis["pattern_violations"] = violations
    analysis["type_averages"] = type_averages

    if violations:
        console.print(f"\n[yellow]{len(violations)} pattern violations found:[/yellow]")
        for v in violations:
            console.print(
                f"  {v['foil_type']}.{v['dimension']}: "
                f"expected {v['expected_range']}, got {v['actual']:.2f}"
            )
    else:
        console.print("\n[green]All expected patterns validated.[/green]")

    return analysis
