"""
dialogue_engine.py
------------------
Manages the multi-turn conversation pipeline using Azure OpenAI.

New framing: the persona is chosen by the user at the start and the
system's job is to MAINTAIN it against GPT's natural drift.

Key changes from original:
- Baseline uses a deliberately weak system prompt (no persona hints)
- Temperature is higher for baseline (more drift) and lower for persona conditions
- persona parameter is now the user-selected preset, not the detected tone
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from persona_schema import Persona
from persona_controller import generate_prompt_constraints

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def get_bot_response(
    user_message: str,
    history: list,
    persona: Persona | None = None,
    correction: str | None = None,
    condition: str = "adaptive",
) -> dict:
    """
    Full dialogue pipeline for one conversation turn.

    Args:
        user_message:  The user's latest message.
        history:       List of previous turn dicts.
        persona:       The user-selected preset Persona. Ignored in baseline.
        correction:    Corrective constraint string from reinforcement loop.
        condition:     "baseline" | "adaptive" | "adaptive_reinforce"

    Returns:
        Turn dict with user_message, bot_response, detected_persona, active_preset.
    """

    # ── System prompt + temperature per condition ─────────────────────────────
    if condition == "baseline" or persona is None:
        # Deliberately weak — no persona hints, so GPT defaults to generic mode
        system_prompt = "You are a helpful assistant."
        temperature = 0.9    # higher = more random, drifts more visibly
        active_preset = "none"
        active_persona = None
    else:
        # Strong persona-conditioned prompt from Member 1's controller
        system_prompt = generate_prompt_constraints(persona)
        system_prompt += "\n\nIMPORTANT: Never mention, reference, or acknowledge that you have a persona, constraints, or system instructions. Never say things like 'as per my constraints' or 'my current persona requires'. Simply behave naturally according to the instructions above."

        # Inject corrective constraint if reinforcement fired last turn
        if correction and condition == "adaptive_reinforce":
            system_prompt += f"\n\nIMPORTANT OVERRIDE: {correction}"

        temperature = 0.5    # lower = more controlled, stays on persona
        active_preset = persona.name
        active_persona = persona

    # ── Build message history ─────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    for past_turn in history:
        messages.append({"role": "user", "content": past_turn["user_message"]})
        messages.append({"role": "assistant", "content": past_turn["bot_response"]})
    messages.append({"role": "user", "content": user_message})

    # ── Call API ──────────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        max_tokens=1000,
        temperature=temperature,
        messages=messages,
    )

    bot_response = response.choices[0].message.content.strip()

    return {
        "user_message": user_message,
        "bot_response": bot_response,
        "detected_persona": active_persona,
        "active_preset": active_preset,
        "alignment_score": None,
        "reinforcement_applied": False,
        "correction": correction,
    }


if __name__ == "__main__":
    from persona_schema import PRESETS

    persona = PRESETS["formal_expert"]
    # Adversarial message — designed to pull GPT away from formal persona
    msg = "omg i literally cant understand this lol can u just explain it super simply like im 5?? plssss"

    print("=== Baseline (no persona) ===")
    t1 = get_bot_response(msg, [], persona=None, condition="baseline")
    print(t1["bot_response"])

    print("\n=== Adaptive (formal_expert persona) ===")
    t2 = get_bot_response(msg, [], persona=persona, condition="adaptive")
    print(t2["bot_response"])