"""
persona_controller.py
---------------------
Maps a Persona object to a concrete, LLM-ready prompt constraint string.
The output of generate_prompt_constraints() is injected as the system
prompt (or prepended to it) in every dialogue turn.
"""

from persona_schema import Persona


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------
def _level(value: float) -> str:
    """Converts a 0-1 float to a low/medium/high label."""
    if value < 0.35:
        return "low"
    if value < 0.65:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Individual trait instruction builders
# ---------------------------------------------------------------------------
def _formality_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Use a casual, conversational tone. Contractions, informal words, and relaxed grammar are all fine.",
        "medium": "Use a balanced tone — professional but not stiff. Avoid slang but also avoid overly formal language.",
        "high":   "Use strictly formal language. No contractions, no slang. Write as if addressing a professional audience.",
    }[level]


def _empathy_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Focus purely on information. Do not comment on the user's emotional state.",
        "medium": "Occasionally acknowledge the user's perspective or feelings where clearly relevant.",
        "high":   (
            "Always acknowledge the user's emotions before providing information. "
            "Use reflective language such as 'I understand that...' or 'That sounds frustrating...'. "
            "Prioritize making the user feel heard."
        ),
    }[level]


def _technical_depth_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Use plain, everyday language. Avoid jargon. Explain any technical terms in simple words.",
        "medium": "Use moderately technical language. Define terms when first introduced but don't over-explain.",
        "high":   (
            "Use precise, domain-specific vocabulary freely. "
            "Assume the user has expert-level knowledge. No need to define standard terms."
        ),
    }[level]


def _verbosity_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Be extremely concise. Respond in one to three sentences maximum unless forced otherwise.",
        "medium": "Aim for moderate length. Cover the key points without padding or unnecessary elaboration.",
        "high":   (
            "Give thorough, detailed responses. Include background context, examples, and edge cases where helpful. "
            "Completeness is more important than brevity."
        ),
    }[level]


def _assertiveness_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Be tentative. Use hedging phrases like 'it might be', 'possibly', or 'you could consider'.",
        "medium": "State views clearly but acknowledge uncertainty where it genuinely exists.",
        "high":   "State conclusions directly and confidently. Avoid unnecessary hedging or qualifications.",
    }[level]


def _humor_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Maintain a completely serious tone at all times. No jokes, wit, or light-heartedness.",
        "medium": "Light humor or wit is acceptable when naturally appropriate, but keep it subtle.",
        "high":   "Feel free to be witty, playful, or humorous when it fits the context. Levity is welcome.",
    }[level]


def _politeness_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Be direct and blunt. No softening phrases or courtesy filler.",
        "medium": "Be courteous but efficient. A brief 'please' or 'thank you' is fine; don't overdo it.",
        "high":   (
            "Be consistently courteous and considerate. Use polite phrases naturally. "
            "Soften any criticism or correction with care."
        ),
    }[level]


def _curiosity_instruction(val: float) -> str:
    level = _level(val)
    return {
        "low":    "Do not ask follow-up questions. Answer what was asked and stop.",
        "medium": "Ask a follow-up question occasionally when it would genuinely help clarify or deepen the conversation.",
        "high":   (
            "Actively express interest in the user's situation. "
            "Ask thoughtful follow-up questions at the end of most responses."
        ),
    }[level]


def _tone_instruction(tone: str) -> str:
    return {
        "neutral":      "Overall tone: balanced and neutral.",
        "warm":         "Overall tone: warm, friendly, and approachable.",
        "authoritative":"Overall tone: authoritative and confident.",
        "playful":      "Overall tone: playful and engaging.",
        "empathetic":   "Overall tone: deeply empathetic and emotionally attuned.",
        "analytical":   "Overall tone: analytical, precise, and objective.",
    }.get(tone, "Overall tone: neutral.")


# ---------------------------------------------------------------------------
# Main controller function
# ---------------------------------------------------------------------------
def generate_prompt_constraints(persona: Persona) -> str:
    """
    Takes a Persona object and returns a formatted system-prompt string
    that instructs the LLM to match the persona's traits.

    This string should be used as (or prepended to) the system prompt
    in every dialogue turn.
    """

    sections = [
        f"You are acting as the '{persona.name}' persona. Follow these behavioral constraints strictly:\n",
        f"[Formality]        {_formality_instruction(persona.formality)}",
        f"[Empathy]          {_empathy_instruction(persona.empathy)}",
        f"[Technical Depth]  {_technical_depth_instruction(persona.technical_depth)}",
        f"[Verbosity]        {_verbosity_instruction(persona.verbosity)}",
        f"[Assertiveness]    {_assertiveness_instruction(persona.assertiveness)}",
        f"[Humor]            {_humor_instruction(persona.humor)}",
        f"[Politeness]       {_politeness_instruction(persona.politeness)}",
        f"[Curiosity]        {_curiosity_instruction(persona.curiosity)}",
        f"[Tone]             {_tone_instruction(persona.tone)}",
        "\nMaintain these traits consistently across every message in the conversation.",
    ]

    return "\n".join(sections)


def generate_corrective_constraint(trait: str, expected_level: str) -> str:
    """
    Returns a corrective instruction for a single trait that has drifted.
    Used by reinforcement loop to patch the next prompt.

    Args:
        trait:          The trait name (must match Persona field names).
        expected_level: 'low', 'medium', or 'high'

    Returns:
        A one-line corrective instruction string.
    """
    corrective_map = {
        "formality": {
            "low":    "CORRECTION: Your last response was too formal. Relax your language immediately.",
            "medium": "CORRECTION: Adjust your formality level to be more balanced — not too casual, not too stiff.",
            "high":   "CORRECTION: Your last response was too casual. Switch to strictly formal language now.",
        },
        "empathy": {
            "low":    "CORRECTION: Too much emotional language. Stick to facts and information only.",
            "medium": "CORRECTION: Rebalance empathy — acknowledge feelings briefly but keep focus on information.",
            "high":   "CORRECTION: You did not acknowledge the user's emotions. Do so explicitly in your next response.",
        },
        "verbosity": {
            "low":    "CORRECTION: Your response was too long. Be much more concise in your next reply.",
            "medium": "CORRECTION: Aim for moderate length — not too short, not too long.",
            "high":   "CORRECTION: Your response was too brief. Provide more detail and depth.",
        },
        "technical_depth": {
            "low":    "CORRECTION: Too much jargon. Use plain language that a non-expert can understand.",
            "medium": "CORRECTION: Adjust technical depth to be moderate — some terms are fine but define new ones.",
            "high":   "CORRECTION: Not technical enough. Use precise domain vocabulary freely.",
        },
        "politeness": {
            "low":    "CORRECTION: Too polite and soft. Be more direct.",
            "medium": "CORRECTION: Rebalance politeness to be courteous but efficient.",
            "high":   "CORRECTION: Your tone was too blunt. Soften your language and be more considerate.",
        },
    }

    trait_map = corrective_map.get(trait)
    if trait_map is None:
        return f"CORRECTION: Recheck your adherence to the '{trait}' trait at level '{expected_level}'."
    return trait_map.get(expected_level, f"CORRECTION: Adjust '{trait}' to '{expected_level}' level.")
