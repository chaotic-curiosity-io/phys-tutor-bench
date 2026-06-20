"""LLM-based simulated student for physics tutoring conversations.

The student simulator is a CONTROLLED VARIABLE — it uses a fixed model, prompt, and
temperature across all evaluations. Changing it invalidates prior results.
"""

from __future__ import annotations

import anthropic
import httpx

from src.scenarios.schema import Scenario, StudentProfile, Affect, ResponseStyle
from src.engine.model_compat import anthropic_create_kwargs, first_text

STUDENT_SYSTEM_PROMPT = """\
You are a simulated introductory physics student in a tutoring session. You must behave \
according to your student profile and misconception consistently throughout the conversation.

## Your Misconception
{misconception_description}

You genuinely believe this misconception is correct. You reason from it naturally. You do \
NOT know you have a misconception — from your perspective, your reasoning makes sense.

## Your Profile
- **Knowledge Level:** {knowledge_level}
- **You already understand:** {knows}
- **You struggle with:** {struggles_with}
- **Your affect:** {affect_description}
- **Your response style:** {style_description}

## Rules for Behavior
1. **Hold your misconception firmly.** Do not be trivially convinced. If the tutor just \
tells you the answer, you may say "okay" but your next response should reveal you haven't \
actually changed your mental model.
2. **Show realistic intermediate states.** As the tutor scaffolds, you may:
   - Ask clarifying questions
   - Show partial understanding mixed with residual confusion
   - Occasionally backslide to your original misconception
   - Express frustration or eureka moments appropriate to your affect
3. **Respond at your knowledge level.** If you're an intro algebra student, don't use \
calculus. If you struggle with vectors, show it.
4. **Be genuine on transfer problems.** When given a new problem, attempt it honestly \
based on your CURRENT understanding at that point in the conversation. If you've genuinely \
shifted your understanding, apply the new concept. If not, apply your misconception again.
5. **Do NOT be a perfect student.** Real students ramble, get distracted, give partial \
answers, and sometimes need things repeated.
6. **Respond in character.** Match the response style — verbose reasoners think out loud, \
terse answerers give short replies, formula pluggers reach for equations, etc.

## The Problem Context
{problem_context}

## Your Initial Response (already given)
You already said: "{initial_response}"

Continue the conversation from here, responding to whatever the tutor says next.
"""

AFFECT_DESCRIPTIONS = {
    Affect.CONFIDENT_BUT_WRONG: "You are confident in your (incorrect) reasoning. You push back when challenged and defend your position.",
    Affect.UNCERTAIN_AND_WRONG: "You are not sure of your answer and know you might be wrong, but your reasoning still comes from the misconception.",
    Affect.PARTIALLY_CORRECT: "You have some correct intuitions mixed with the misconception. You sense something is off but can't pinpoint it.",
    Affect.CONFUSED_AND_FRUSTRATED: "You are confused and getting frustrated. You feel like physics doesn't make sense.",
    Affect.DISENGAGED: "You are going through the motions but aren't really invested. Short, minimal responses unless something catches your interest.",
    Affect.EAGER_BUT_WRONG: "You are enthusiastic and eager to learn but your reasoning is wrong. You respond positively to the tutor's guidance.",
}

STYLE_DESCRIPTIONS = {
    ResponseStyle.VERBOSE_REASONER: "You think out loud, explaining your reasoning in detail. You connect ideas as you talk.",
    ResponseStyle.TERSE_ANSWERER: "You give short, direct answers. You don't elaborate unless asked.",
    ResponseStyle.QUESTION_ASKER: "You respond to explanations with more questions. You like to understand 'why' and 'how'.",
    ResponseStyle.FORMULA_PLUGGER: "You reach for equations and formulas. You try to solve things numerically even when conceptual reasoning would be better.",
    ResponseStyle.HEDGING_GUESSER: "You hedge your answers with 'I think...', 'maybe...', 'I'm not sure but...'. You float tentative ideas.",
}


def build_student_system_prompt(scenario: Scenario) -> str:
    """Construct the student simulator system prompt from a scenario."""
    profile = scenario.student_profile

    return STUDENT_SYSTEM_PROMPT.format(
        misconception_description=scenario.misconception_description,
        knowledge_level=profile.knowledge_level.value,
        knows=", ".join(profile.knows),
        struggles_with=", ".join(profile.struggles_with),
        affect_description=AFFECT_DESCRIPTIONS.get(profile.affect, str(profile.affect)),
        style_description=STYLE_DESCRIPTIONS.get(profile.response_style, str(profile.response_style)),
        problem_context=scenario.problem_context,
        initial_response=scenario.student_initial_response,
    )


class StudentSimulator:
    """Simulates a physics student with specified misconceptions.

    Uses a FIXED model and temperature as a controlled variable.
    """

    def __init__(
        self,
        scenario: Scenario,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        client: anthropic.Anthropic | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.scenario = scenario
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # When api_base is set, the student is served by an OpenAI-compatible endpoint
        # (e.g. a local Ollama server) instead of the Anthropic SDK.
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key = api_key
        self.timeout = timeout
        self.client = client or (None if self.api_base else anthropic.Anthropic())
        self.system_prompt = build_student_system_prompt(scenario)

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        inject_transfer: bool = False,
    ) -> tuple[str, int]:
        """Generate a student response given the conversation history.

        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": str}
                where "user" = tutor messages and "assistant" = student messages.
            inject_transfer: If True, the student will introduce the transfer problem.

        Returns:
            Tuple of (response_text, token_count).
        """
        system = self.system_prompt
        if inject_transfer:
            system += (
                f"\n\n## IMPORTANT: Transfer Problem Injection\n"
                f"In your next response, after responding to the tutor, say something like:\n"
                f"\"Okay, I think I'm starting to get it. But what about this — "
                f"{self.scenario.transfer_problem}\"\n"
                f"Phrase it naturally in your own words and response style."
            )

        if self.api_base:
            # OpenAI-compatible endpoint (Ollama): fold the system prompt in as the
            # first message, since this API takes system as a role rather than a kwarg.
            messages = [{"role": "system", "content": system}] + list(conversation_history)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            resp = httpx.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            return text, tokens

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=conversation_history,
            **anthropic_create_kwargs(self.model, self.temperature, self.max_tokens),
        )

        text = first_text(response.content)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens
