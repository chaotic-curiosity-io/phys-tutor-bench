"""Generate physics tutoring scenarios from the misconception taxonomy using the Anthropic API."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic
from pydantic import ValidationError

from src.scenarios.schema import Scenario, TopicArea
from src.scenarios.taxonomy import MisconceptionTaxonomy, MisconceptionEntry

GENERATION_PROMPT = """\
You are a Physics Education Research (PER) expert creating tutoring benchmark scenarios.

Given a specific misconception from a validated PER instrument, generate a realistic tutoring scenario where a student exhibits this misconception.

## Misconception Details
- **ID:** {misconception_id}
- **Description:** {description}
- **Topic Area:** {topic_area}
- **Subtopic:** {subtopic}
- **Source Instrument:** {source_instrument}
- **Typical Student Reasoning:** {typical_reasoning}
- **Related Concepts:** {related_concepts}

## Requirements
1. The problem_context should be a concrete, physically grounded situation (not abstract).
2. The student_initial_response must naturally exhibit the specified misconception. It should sound like a real intro physics student — not perfectly articulate, sometimes rambling, but with clear reasoning that reveals the misconception.
3. The transfer_problem must test the SAME concept in a DIFFERENT physical context. It should be far enough from the original that rote memorization won't help.
4. The expert_strategy should reflect PER best practices: bridging analogies, anchoring conceptions, scaffolding from what the student already knows.
5. The student_profile should be internally consistent with the initial response.

## Scenario Number
This is scenario {scenario_number} of {total_for_misconception} for this misconception. Vary the:
- Physical context (different real-world situations)
- Student profile (different affects, response styles, knowledge levels)
- Transfer problem (different application domains)

## Output Format
Return ONLY a valid JSON object matching this exact schema (no markdown, no extra text):

{{
  "id": "{scenario_id}",
  "topic_area": "{topic_area_enum}",
  "subtopic": "{subtopic}",
  "level": "<intro_algebra|intro_calculus|upper_division>",
  "problem_context": "<concrete physics problem>",
  "student_initial_response": "<realistic student response exhibiting the misconception>",
  "misconception_tags": ["{misconception_id}"],
  "misconception_source": "{source_instrument}",
  "misconception_description": "{description}",
  "common_prevalence": "{prevalence}",
  "student_profile": {{
    "knowledge_level": "<intro_algebra|intro_calculus|upper_division>",
    "knows": ["<concept1>", "<concept2>"],
    "struggles_with": ["<concept1>", "<concept2>"],
    "affect": "<confident_but_wrong|uncertain_and_wrong|partially_correct|confused_and_frustrated|disengaged|eager_but_wrong>",
    "response_style": "<verbose_reasoner|terse_answerer|question_asker|formula_plugger|hedging_guesser>"
  }},
  "transfer_problem": "<novel problem requiring same concept>",
  "transfer_success_criteria": "<what successful transfer looks like>",
  "expert_strategy": {{
    "diagnosis": "<concise misconception diagnosis>",
    "recommended_approach": "<pedagogical approach>",
    "scaffolding_sequence": ["<step1>", "<step2>", "<step3>", "<step4>", "<step5>"],
    "common_pitfalls": ["<pitfall1>", "<pitfall2>", "<pitfall3>"]
  }}
}}
"""

# How many scenarios to generate per misconception, by topic area target
TOPIC_TARGETS = {
    "mechanics": 100,
    "em": 60,
    "thermal_waves": 40,
    "modern_quantum": 30,
}


def _make_scenario_id(topic_area: str, subtopic: str, misconception_id: str, number: int) -> str:
    """Generate a scenario ID like mech-n3l-ar-001."""
    topic_prefix = {
        "mechanics": "mech",
        "em": "em",
        "thermal_waves": "thwv",
        "modern_quantum": "modq",
    }
    prefix = topic_prefix.get(topic_area, topic_area[:4])
    # Shorten subtopic
    sub = subtopic.replace("_", "")[:4]
    # Shorten misconception
    misc = misconception_id.lower().replace("_", "")[:6]
    return f"{prefix}-{sub}-{misc}-{number:03d}"


def _topic_area_to_enum(topic_area: str) -> str:
    """Map topic area string to TopicArea enum value."""
    return topic_area


def generate_scenario(
    client: anthropic.Anthropic,
    entry: MisconceptionEntry,
    scenario_number: int,
    total_for_misconception: int,
    model: str = "claude-sonnet-4-20250514",
) -> Scenario | None:
    """Generate a single scenario for a given misconception entry."""
    scenario_id = _make_scenario_id(
        entry.topic_area, entry.subtopic, entry.id, scenario_number
    )

    prompt = GENERATION_PROMPT.format(
        misconception_id=entry.id,
        description=entry.description,
        topic_area=entry.topic_area,
        subtopic=entry.subtopic,
        source_instrument=entry.source_instrument or "PER literature",
        typical_reasoning=entry.typical_student_reasoning,
        related_concepts=", ".join(entry.related_concepts),
        scenario_number=scenario_number,
        total_for_misconception=total_for_misconception,
        scenario_id=scenario_id,
        topic_area_enum=_topic_area_to_enum(entry.topic_area),
        prevalence=entry.prevalence.value,
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Extract JSON from response (handle potential markdown wrapping)
    json_match = re.search(r'\{[\s\S]*\}', raw_text)
    if not json_match:
        print(f"  WARNING: No JSON found in response for {scenario_id}")
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error for {scenario_id}: {e}")
        return None

    try:
        scenario = Scenario(**data)
    except ValidationError as e:
        print(f"  WARNING: Validation error for {scenario_id}: {e}")
        return None

    return scenario


def compute_scenarios_per_misconception(taxonomy: MisconceptionTaxonomy) -> dict[str, int]:
    """Compute how many scenarios to generate per misconception to hit topic targets."""
    allocation: dict[str, int] = {}

    for topic_area, target in TOPIC_TARGETS.items():
        entries = taxonomy.by_topic(topic_area)
        if not entries:
            continue
        per_entry = max(1, target // len(entries))
        remainder = target - (per_entry * len(entries))
        for i, entry in enumerate(entries):
            extra = 1 if i < remainder else 0
            allocation[entry.id] = per_entry + extra

    return allocation


def generate_all_scenarios(
    taxonomy_path: str | Path,
    output_dir: str | Path,
    topic_filter: str | None = None,
    count_override: int | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> list[Scenario]:
    """Generate scenarios for all misconceptions in the taxonomy.

    Args:
        taxonomy_path: Path to misconception_taxonomy.json
        output_dir: Base output directory (e.g., data/scenarios)
        topic_filter: Optional topic area to filter (e.g., "mechanics")
        count_override: Override per-misconception count
        model: Anthropic model to use for generation
    """
    taxonomy = MisconceptionTaxonomy(taxonomy_path)
    output_dir = Path(output_dir)
    client = anthropic.Anthropic()

    allocation = compute_scenarios_per_misconception(taxonomy)
    entries = taxonomy.all_entries

    if topic_filter:
        entries = [e for e in entries if e.topic_area == topic_filter]

    all_scenarios: list[Scenario] = []

    for entry in entries:
        n_scenarios = count_override or allocation.get(entry.id, 2)
        print(f"Generating {n_scenarios} scenario(s) for {entry.id} ({entry.topic_area}/{entry.subtopic})")

        for i in range(1, n_scenarios + 1):
            scenario = generate_scenario(client, entry, i, n_scenarios, model=model)
            if scenario is None:
                continue

            all_scenarios.append(scenario)

            # Write to topic-specific directory
            topic_dir_map = {
                "mechanics": "mechanics",
                "em": "em",
                "thermal_waves": "thermal_waves",
                "modern_quantum": "modern_quantum",
            }
            topic_dir = output_dir / topic_dir_map.get(entry.topic_area, entry.topic_area)
            topic_dir.mkdir(parents=True, exist_ok=True)

            out_path = topic_dir / f"{scenario.id}.json"
            out_path.write_text(json.dumps(scenario.model_dump(), indent=2))
            print(f"  -> {out_path}")

    print(f"\nGenerated {len(all_scenarios)} scenarios total.")
    return all_scenarios
