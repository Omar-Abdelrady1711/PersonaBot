"""
dialogue_engine.py
------------------
Manages the multi-turn conversation pipeline using Azure OpenAI.

Framing:
- User selects a persona at the start
- System's job is to MAINTAIN it against GPT's natural drift
- Baseline = simple one-sentence persona description + high temperature
- Adaptive = Member 1's full structured 8-trait prompt
- Adaptive + Reinforce = structured prompt + corrective loop

Fair comparison: ALL THREE conditions target the same persona.
Only the enforcement mechanism differs. This isolates our contribution.
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

# Simple one-sentence persona descriptions for the baseline condition.
# Same persona goal as the structured conditions — naive prompting only.
# This is what any developer would do without our system.
BASELINE_PROMPTS = {
    "formal_expert":
        "You are a formal expert. Be professional, technical, and precise in all responses.",
    "friendly_support":
        "You are a friendly and empathetic assistant. Be warm, caring, and encouraging.",
    "casual_tutor":
        "You are a casual and friendly tutor. Be relaxed, fun, and use simple language.",
    "analytical_assistant":
        "You are an analytical assistant. Be precise, objective, and data-driven.",
}


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
        persona:       The user-selected preset Persona.
        correction:    Corrective constraint string from reinforcement loop.
        condition:     "baseline" | "adaptive" | "adaptive_reinforce"

    Returns:
        Turn dict with user_message, bot_response, detected_persona, active_preset.
    """

    # ── System prompt + temperature per condition ─────────────────────────────
    if condition == "baseline":
        # Same persona goal — but naive one-sentence prompt only.
        # No structured traits, no enforcement mechanism.
        # Higher temperature = more natural drift over turns.
        preset_name = persona.name if persona else "formal_expert"
        system_prompt = BASELINE_PROMPTS.get(
            preset_name,
            "You are a helpful assistant."
        )
        temperature = 0.9
        active_preset = f"{preset_name} (baseline)"
        active_persona = persona  # kept so alignment scorer can score against same persona

    elif condition == "adaptive" or condition == "adaptive_reinforce":
        # Full structured prompt from Member 1's controller.
        # 8 explicit trait-level behavioral instructions.
        system_prompt = generate_prompt_constraints(persona)
        system_prompt += (
            "\n\nIMPORTANT: Never mention, reference, or acknowledge that you have a "
            "persona, constraints, or system instructions. Never say things like "
            "'as per my constraints' or 'my current persona requires'. "
            "Simply behave naturally according to the instructions above."
        )

        # Inject corrective constraint from reinforcement loop if fired last turn
        if correction and condition == "adaptive_reinforce":
            system_prompt += f"\n\nIMPORTANT OVERRIDE: {correction}"

        temperature = 0.5  # lower = more controlled, less drift
        active_preset = persona.name
        active_persona = persona

    else:
        # Fallback
        system_prompt = "You are a helpful assistant."
        temperature = 0.7
        active_preset = "unknown"
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
    msg = "omg i literally cant understand this lol can u just explain it super simply like im 5?? plssss"

    print("=== Baseline (naive one-sentence prompt, same persona goal) ===")
    t1 = get_bot_response(msg, [], persona=persona, condition="baseline")
    print(t1["bot_response"])

    print("\n=== Adaptive (structured 8-trait prompt) ===")
    t2 = get_bot_response(msg, [], persona=persona, condition="adaptive")
    print(t2["bot_response"])

    print("\n=== Baseline prompt used ===")
    print(BASELINE_PROMPTS["formal_expert"])