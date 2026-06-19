"""Orchestrates the multi-turn tutoring conversation between tutor and student."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.engine.student_simulator import StudentSimulator
from src.engine.tutor_runner import TutorBackend, DEFAULT_TUTOR_SYSTEM_PROMPT
from src.scenarios.schema import Scenario, ConversationRecord, ConversationMessage


class ConversationLoop:
    """Runs a single tutoring conversation for a given scenario."""

    def __init__(
        self,
        scenario: Scenario,
        tutor: TutorBackend,
        student: StudentSimulator,
        max_turns: int = 15,
        transfer_injection_offset: int = 2,
        tutor_system_prompt: str = DEFAULT_TUTOR_SYSTEM_PROMPT,
    ):
        self.scenario = scenario
        self.tutor = tutor
        self.student = student
        self.max_turns = max_turns
        self.transfer_injection_offset = transfer_injection_offset
        self.tutor_system_prompt = tutor_system_prompt

    def run(self) -> ConversationRecord:
        """Execute the full tutoring conversation and return the record."""
        conversation_id = str(uuid4())
        start_time = datetime.now(timezone.utc)

        messages: list[ConversationMessage] = []
        # The "history" from the tutor's perspective: student messages are "user", tutor messages are "assistant"
        tutor_history: list[dict[str, str]] = []
        # The "history" from the student's perspective: tutor messages are "user", student messages are "assistant"
        student_history: list[dict[str, str]] = []

        total_tutor_tokens = 0
        total_student_tokens = 0
        termination_reason = "max_turns"
        turn = 0
        transfer_injected = False

        # Turn 0: Student's initial response (from the scenario)
        initial_msg = ConversationMessage(
            role="student",
            content=self.scenario.student_initial_response,
            turn_number=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        messages.append(initial_msg)

        # Seed tutor history with the problem context + student's initial response
        problem_intro = (
            f"The student is working on this problem:\n\n"
            f"{self.scenario.problem_context}\n\n"
            f"The student says:\n\n"
            f"{self.scenario.student_initial_response}"
        )
        tutor_history.append({"role": "user", "content": problem_intro})

        for turn in range(1, self.max_turns + 1):
            # --- Tutor responds ---
            tutor_text, tutor_tokens = self.tutor.respond(
                tutor_history, self.tutor_system_prompt
            )
            total_tutor_tokens += tutor_tokens

            tutor_msg = ConversationMessage(
                role="tutor",
                content=tutor_text,
                turn_number=turn,
                timestamp=datetime.now(timezone.utc).isoformat(),
                token_count=tutor_tokens,
            )
            messages.append(tutor_msg)

            # Update histories
            tutor_history.append({"role": "assistant", "content": tutor_text})
            student_history.append({"role": "user", "content": tutor_text})

            # Check if we've hit max turns (tutor gets last word)
            if turn >= self.max_turns:
                termination_reason = "max_turns"
                break

            # --- Student responds ---
            # Pose the transfer problem deterministically as the student's message the
            # first time we reach the final window. Relying on the student model to
            # introduce it via a prompt instruction proved unreliable for local models,
            # so we inject the exact transfer problem to guarantee it is probed. The
            # student's genuine (mis)understanding then shows in how it ATTEMPTS the
            # transfer over the following turns.
            in_transfer_window = turn >= self.max_turns - self.transfer_injection_offset

            if in_transfer_window and not transfer_injected:
                student_text = (
                    "Okay, I think I'm starting to get it. But what about this — "
                    f"{self.scenario.transfer_problem}"
                )
                student_tokens = 0
                transfer_injected = True
            else:
                student_text, student_tokens = self.student.respond(
                    student_history, inject_transfer=False
                )
                total_student_tokens += student_tokens

            student_msg = ConversationMessage(
                role="student",
                content=student_text,
                turn_number=turn,
                timestamp=datetime.now(timezone.utc).isoformat(),
                token_count=student_tokens,
            )
            messages.append(student_msg)

            # Update histories
            student_history.append({"role": "assistant", "content": student_text})
            tutor_history.append({"role": "user", "content": student_text})

            # Check for student disengagement (heuristic)
            if self._check_disengagement(student_text):
                termination_reason = "student_disengaged"
                break

        record = ConversationRecord(
            id=conversation_id,
            scenario_id=self.scenario.id,
            model_under_test=getattr(self.tutor, 'model', 'unknown'),
            student_simulator_model=self.student.model,
            student_simulator_temperature=self.student.temperature,
            messages=messages,
            total_turns=turn,
            termination_reason=termination_reason,
            tutor_system_prompt=self.tutor_system_prompt,
            student_system_prompt=self.student.system_prompt,
            run_timestamp=start_time.isoformat(),
            total_tutor_tokens=total_tutor_tokens,
            total_student_tokens=total_student_tokens,
        )

        return record

    def _check_disengagement(self, text: str) -> bool:
        """Heuristic check for student disengagement signals."""
        disengagement_phrases = [
            "i give up",
            "i don't care",
            "this is pointless",
            "whatever",
            "i quit",
            "forget it",
            "never mind",
        ]
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in disengagement_phrases)


def save_conversation(record: ConversationRecord, output_dir: str | Path) -> Path:
    """Save a conversation record to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record.id}.json"
    path.write_text(json.dumps(record.model_dump(), indent=2))
    return path


def load_conversation(path: str | Path) -> ConversationRecord:
    """Load a conversation record from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return ConversationRecord(**data)
