"""
persona_schema.py
-----------------
Defines the Persona data structure used throughout the dialogue system.
Each trait is a float in [0.0, 1.0] unless stated otherwise.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


# ---------------------------------------------------------------------------
# Trait ranges and documentation
# ---------------------------------------------------------------------------
TRAIT_DOCS = {
    "formality": (
        "How formal the language should be. "
        "0.0 = very casual/slang-friendly, 1.0 = strictly professional."
    ),
    "empathy": (
        "How much the bot acknowledges emotions and feelings. "
        "0.0 = purely informational, 1.0 = highly empathetic and emotionally aware."
    ),
    "technical_depth": (
        "How technical the vocabulary and explanations should be. "
        "0.0 = plain language for a general audience, 1.0 = expert-level domain vocabulary."
    ),
    "verbosity": (
        "How long and detailed the responses should be. "
        "0.0 = extremely terse (one-liners), 1.0 = exhaustive and thorough."
    ),
    "assertiveness": (
        "How directly and confidently opinions are stated. "
        "0.0 = very tentative/hedging, 1.0 = direct and definitive."
    ),
    "humor": (
        "How much light-heartedness or wit is allowed. "
        "0.0 = completely serious, 1.0 = frequent humor and wit."
    ),
    "politeness": (
        "How courteous and deferential the tone is. "
        "0.0 = blunt with no softening, 1.0 = extremely polite and considerate."
    ),
    "curiosity": (
        "How much the bot asks follow-up questions and expresses interest. "
        "0.0 = never asks questions, 1.0 = actively curious and probing."
    ),
}

VALID_TONES = Literal["neutral", "warm", "authoritative", "playful", "empathetic", "analytical"]


# ---------------------------------------------------------------------------
# Persona dataclass
# ---------------------------------------------------------------------------
@dataclass
class Persona:
    """
    Structured representation of a chatbot persona.

    All float traits are in the range [0.0, 1.0].
    tone is a categorical label that works alongside the numeric traits.
    name is a human-readable label for logging and debugging.
    """

    name: str

    # Numeric traits
    formality:       float = 0.5
    empathy:         float = 0.5
    technical_depth: float = 0.5
    verbosity:       float = 0.5
    assertiveness:   float = 0.5
    humor:           float = 0.2
    politeness:      float = 0.7
    curiosity:       float = 0.4

    # Categorical override
    tone: str = "neutral"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        float_traits = [
            "formality", "empathy", "technical_depth", "verbosity",
            "assertiveness", "humor", "politeness", "curiosity",
        ]
        for trait in float_traits:
            val = getattr(self, trait)
            if not isinstance(val, (int, float)):
                raise TypeError(f"Trait '{trait}' must be a number, got {type(val)}.")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Trait '{trait}' must be in [0.0, 1.0], got {val}.")

        valid_tones = {"neutral", "warm", "authoritative", "playful", "empathetic", "analytical"}
        if self.tone not in valid_tones:
            raise ValueError(f"tone must be one of {valid_tones}, got '{self.tone}'.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Persona":
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Preset personas for quick use and testing
# ---------------------------------------------------------------------------
PRESETS: dict[str, Persona] = {

    "formal_expert": Persona(
        name="formal_expert",
        formality=0.9,
        empathy=0.2,
        technical_depth=0.9,
        verbosity=0.7,
        assertiveness=0.8,
        humor=0.0,
        politeness=0.6,
        curiosity=0.3,
        tone="authoritative",
    ),

    "friendly_support": Persona(
        name="friendly_support",
        formality=0.2,
        empathy=0.9,
        technical_depth=0.2,
        verbosity=0.5,
        assertiveness=0.3,
        humor=0.4,
        politeness=0.9,
        curiosity=0.7,
        tone="warm",
    ),

    "casual_tutor": Persona(
        name="casual_tutor",
        formality=0.3,
        empathy=0.6,
        technical_depth=0.5,
        verbosity=0.6,
        assertiveness=0.5,
        humor=0.5,
        politeness=0.7,
        curiosity=0.8,
        tone="playful",
    ),

    "analytical_assistant": Persona(
        name="analytical_assistant",
        formality=0.7,
        empathy=0.1,
        technical_depth=0.8,
        verbosity=0.8,
        assertiveness=0.7,
        humor=0.0,
        politeness=0.5,
        curiosity=0.5,
        tone="analytical",
    ),
}
