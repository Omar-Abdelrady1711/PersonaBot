"""
dialogue_engine.py
------------------
Manages the multi-turn conversation pipeline using Azure OpenAI.
Calls Member 1's controller to build the system prompt,
injects corrections from Member 3's reinforcement loop,
and returns a structured turn dict.
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
        history:       List of previous turn dicts (from session state).
        persona:       Active Persona object. Ignored in baseline condition.
        correction:    Corrective constraint string from Member 3's reinforcement loop.
        condition:     "baseline" | "adaptive" | "adaptive_reinforce"

    Returns:
        Turn dict with user_message, bot_response, detected_persona, active_preset.
        Member 3 fills alignment_score, reinforcement_applied, correction fields.
    """

    # ── Build system prompt ───────────────────────────────────────────────────
    if condition == "baseline" or persona is None:
        system_prompt = "You are a helpful assistant."
        active_preset = "none"
        detected_persona = None
    else:
        system_prompt = generate_prompt_constraints(persona)

        # Inject corrective constraint from Member 3 if reinforcement fired
        if correction and condition == "adaptive_reinforce":
            system_prompt += f"\n\n{correction}"

        active_preset = getattr(persona, "name", "unknown")
        detected_persona = persona

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
        messages=messages,
    )

    bot_response = response.choices[0].message.content.strip()

    # ── Return turn dict ──────────────────────────────────────────────────────
    return {
        "user_message": user_message,
        "bot_response": bot_response,
        "detected_persona": detected_persona,
        "active_preset": active_preset,
        "alignment_score": None,         # filled by Member 3
        "reinforcement_applied": False,  # filled by Member 3
        "correction": correction,        # passed through for display
    }


if __name__ == "__main__":
    from tone_detector import detect_tone
    from preset_matcher import match_preset

    msg = "I'm really stressed about my exam tomorrow, what should I do?"
    print(f"User: {msg}\n")

    detected = detect_tone(msg)
    preset_name, preset = match_preset(detected)
    print(f"Detected tone: {detected.tone} → Matched preset: {preset_name}\n")

    turn = get_bot_response(
        user_message=msg,
        history=[],
        persona=preset,
        condition="adaptive"
    )
    print(f"Bot ({preset_name}): {turn['bot_response']}")
