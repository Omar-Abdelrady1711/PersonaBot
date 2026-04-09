"""
preset_matcher.py
-----------------
Maps a detected user Persona to the closest preset from Member 1's PRESETS.
Uses euclidean distance across all 8 traits.
"""

import math
from persona_schema import Persona, PRESETS

TRAIT_KEYS = [
    "formality", "empathy", "technical_depth", "verbosity",
    "assertiveness", "humor", "politeness", "curiosity"
]


def _euclidean_distance(a: Persona, b: Persona) -> float:
    """Computes distance between two personas across all 8 traits."""
    return math.sqrt(sum(
        (getattr(a, t) - getattr(b, t)) ** 2
        for t in TRAIT_KEYS
    ))


def match_preset(detected: Persona) -> tuple[str, Persona]:
    """
    Finds the closest preset persona to the detected user tone.
    Returns (preset_name, preset_persona).
    """
    best_name = None
    best_persona = None
    best_distance = float("inf")

    for name, preset in PRESETS.items():
        dist = _euclidean_distance(detected, preset)
        if dist < best_distance:
            best_distance = dist
            best_name = name
            best_persona = preset

    return best_name, best_persona


def smooth_persona(current: Persona, detected: Persona, alpha: float = 0.3) -> Persona:
    """
    Smoothly transitions the active persona toward the newly detected one.
    Prevents jarring personality whiplash between turns.
    Formula: new = (1 - alpha) * current + alpha * detected
    """
    def blend(a, b):
        return round((1 - alpha) * a + alpha * b, 4)

    return Persona(
        name=current.name,
        formality=blend(current.formality, detected.formality),
        empathy=blend(current.empathy, detected.empathy),
        technical_depth=blend(current.technical_depth, detected.technical_depth),
        verbosity=blend(current.verbosity, detected.verbosity),
        assertiveness=blend(current.assertiveness, detected.assertiveness),
        humor=blend(current.humor, detected.humor),
        politeness=blend(current.politeness, detected.politeness),
        curiosity=blend(current.curiosity, detected.curiosity),
        tone=detected.tone,
    )


if __name__ == "__main__":
    from tone_detector import detect_tone

    test_messages = [
        "hey i'm kinda lost on this lol",
        "I require a precise technical breakdown of attention mechanisms.",
        "I'm so stressed I don't even know where to begin.",
        "Could you perhaps help me understand this concept?",
    ]
    for msg in test_messages:
        detected = detect_tone(msg)
        name, preset = match_preset(detected)
        print(f"\nMessage: {msg}")
        print(f"  Detected tone: {detected.tone}")
        print(f"  Matched preset: {name}")
