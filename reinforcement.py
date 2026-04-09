"""
reinforcement.py
----------------
Checks alignment scores and fires corrective constraints when drift is detected.
Uses Member 1's generate_corrective_constraint() to produce targeted corrections.

Member 3 deliverable — depends only on Member 1's persona_controller.py.
"""

from persona_schema import Persona
from persona_controller import generate_corrective_constraint, _level
from alignment_scorer import score_alignment

# Traits that have corrective support in Member 1's controller
CORRECTABLE_TRAITS = {"formality", "empathy", "verbosity", "technical_depth", "politeness"}

# Default threshold — if overall alignment drops below this, reinforcement fires
DEFAULT_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def apply_reinforcement(turn: dict, threshold: float = DEFAULT_THRESHOLD) -> str | None:
    """
    Checks whether the bot's response aligns with the active persona.
    If overall alignment is below the threshold, generates a corrective constraint
    targeting the worst-performing correctable trait.

    Args:
        turn:      A turn dict. Must contain 'bot_response' and 'detected_persona'.
                   (alignment_score may already be filled in, or this function will compute it.)
        threshold: Minimum acceptable overall alignment score. Default 0.65.

    Returns:
        A corrective constraint string (to inject into the next prompt), or None.
    """
    bot_response = turn.get("bot_response", "")
    persona: Persona | None = turn.get("detected_persona")

    if not bot_response or persona is None:
        return None

    # Use pre-computed alignment scores if available, otherwise compute now
    existing_scores = turn.get("alignment_score")
    if isinstance(existing_scores, dict):
        scores = existing_scores
    else:
        scores = score_alignment(bot_response, persona)

    overall = scores.get("overall", 1.0)

    if overall >= threshold:
        return None  # Alignment is fine — no correction needed

    # Find the worst-performing trait among those we can correct
    correctable_scores = {
        trait: scores[trait]
        for trait in CORRECTABLE_TRAITS
        if trait in scores
    }

    if not correctable_scores:
        return None

    worst_trait = min(correctable_scores, key=correctable_scores.get)
    expected_value = getattr(persona, worst_trait, 0.5)
    expected_level = _level(expected_value)

    correction = generate_corrective_constraint(worst_trait, expected_level)
    return correction


def score_and_reinforce(turn: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Convenience function that:
    1. Scores the bot response against the active persona.
    2. Runs the reinforcement check.
    3. Fills in alignment_score, reinforcement_applied, and correction in the turn dict.

    This is the main function to call after each bot turn.

    Args:
        turn:      Turn dict from Member 2's get_bot_response().
        threshold: Alignment threshold for triggering reinforcement.

    Returns:
        The same turn dict with alignment_score, reinforcement_applied, correction filled in.
    """
    bot_response = turn.get("bot_response", "")
    persona: Persona | None = turn.get("detected_persona")

    # Score alignment
    if bot_response and persona:
        scores = score_alignment(bot_response, persona)
    else:
        scores = {"overall": 1.0}

    turn["alignment_score"] = scores

    # Apply reinforcement
    correction = apply_reinforcement(turn, threshold=threshold)
    turn["reinforcement_applied"] = correction is not None
    turn["correction"] = correction

    return turn


def get_worst_trait(scores: dict) -> str | None:
    """
    Returns the name of the worst-performing correctable trait.
    Useful for display in the Streamlit UI.
    """
    correctable_scores = {
        trait: scores[trait]
        for trait in CORRECTABLE_TRAITS
        if trait in scores
    }
    if not correctable_scores:
        return None
    return min(correctable_scores, key=correctable_scores.get)


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from persona_schema import PRESETS

    # Simulate a turn where the bot responds casually but the active persona is formal_expert
    mismatched_turn = {
        "user_message": "What is attention in transformers?",
        "bot_response": "hey so basically attention is just the model figuring out which words matter lol, pretty cool right?",
        "detected_persona": PRESETS["formal_expert"],
        "active_preset": "formal_expert",
        "alignment_score": None,
        "reinforcement_applied": False,
        "correction": None,
    }

    result = score_and_reinforce(mismatched_turn)
    print("=== Mismatched Turn (casual response vs formal_expert persona) ===")
    print(f"Overall alignment: {result['alignment_score']['overall']:.3f}")
    print(f"Reinforcement fired: {result['reinforcement_applied']}")
    print(f"Correction: {result['correction']}")

    # Simulate a well-matched turn
    matched_turn = {
        "user_message": "I'm feeling really stressed about everything",
        "bot_response": (
            "I completely understand — it sounds like you're carrying a lot right now, "
            "and that's genuinely tough. You're not alone in feeling this way. "
            "What's been weighing on you the most? I'd really like to help."
        ),
        "detected_persona": PRESETS["friendly_support"],
        "active_preset": "friendly_support",
        "alignment_score": None,
        "reinforcement_applied": False,
        "correction": None,
    }

    result2 = score_and_reinforce(matched_turn)
    print("\n=== Well-Matched Turn (warm response vs friendly_support persona) ===")
    print(f"Overall alignment: {result2['alignment_score']['overall']:.3f}")
    print(f"Reinforcement fired: {result2['reinforcement_applied']}")
    print(f"Correction: {result2['correction']}")