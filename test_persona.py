"""
test_persona.py
---------------
Unit tests for persona_schema.py and persona_controller.py
Run with: pytest test_persona.py -v
"""

import pytest
import json
from persona_schema import Persona, PRESETS, TRAIT_DOCS
from persona_controller import (
    generate_prompt_constraints,
    generate_corrective_constraint,
    _level,
)


# ===========================================================================
# SECTION 1: Persona Schema Tests
# ===========================================================================

class TestPersonaDefaults:
    def test_default_values_in_range(self):
        p = Persona(name="test")
        for trait in ["formality", "empathy", "technical_depth", "verbosity",
                      "assertiveness", "humor", "politeness", "curiosity"]:
            assert 0.0 <= getattr(p, trait) <= 1.0

    def test_default_tone_is_valid(self):
        p = Persona(name="test")
        assert p.tone == "neutral"

    def test_name_stored_correctly(self):
        p = Persona(name="my_bot")
        assert p.name == "my_bot"


class TestPersonaValidation:
    def test_trait_above_1_raises(self):
        with pytest.raises(ValueError, match="formality"):
            Persona(name="bad", formality=1.5)

    def test_trait_below_0_raises(self):
        with pytest.raises(ValueError, match="empathy"):
            Persona(name="bad", empathy=-0.1)

    def test_trait_exactly_0_is_valid(self):
        p = Persona(name="edge", humor=0.0)
        assert p.humor == 0.0

    def test_trait_exactly_1_is_valid(self):
        p = Persona(name="edge", politeness=1.0)
        assert p.politeness == 1.0

    def test_invalid_tone_raises(self):
        with pytest.raises(ValueError, match="tone"):
            Persona(name="bad", tone="aggressive")

    def test_non_numeric_trait_raises(self):
        with pytest.raises(TypeError):
            Persona(name="bad", formality="high")

    @pytest.mark.parametrize("tone", ["neutral", "warm", "authoritative", "playful", "empathetic", "analytical"])
    def test_all_valid_tones_accepted(self, tone):
        p = Persona(name="t", tone=tone)
        assert p.tone == tone


class TestPersonaSerialization:
    def test_to_dict_contains_all_traits(self):
        p = Persona(name="s")
        d = p.to_dict()
        assert "formality" in d
        assert "empathy" in d
        assert "tone" in d
        assert "name" in d

    def test_to_json_is_valid_json(self):
        p = Persona(name="s")
        parsed = json.loads(p.to_json())
        assert parsed["name"] == "s"

    def test_from_dict_roundtrip(self):
        p = Persona(name="roundtrip", formality=0.8, empathy=0.3, tone="warm")
        p2 = Persona.from_dict(p.to_dict())
        assert p2.name == p.name
        assert p2.formality == p.formality
        assert p2.tone == p.tone

    def test_from_json_roundtrip(self):
        p = Persona(name="json_test", technical_depth=0.9)
        p2 = Persona.from_json(p.to_json())
        assert p2.technical_depth == p.technical_depth

    def test_from_dict_invalid_tone_raises(self):
        data = Persona(name="x").to_dict()
        data["tone"] = "rude"
        with pytest.raises(ValueError):
            Persona.from_dict(data)


class TestPresets:
    def test_all_presets_are_valid_personas(self):
        for name, persona in PRESETS.items():
            assert isinstance(persona, Persona)
            assert persona.name == name

    def test_formal_expert_high_formality(self):
        assert PRESETS["formal_expert"].formality >= 0.8

    def test_friendly_support_high_empathy(self):
        assert PRESETS["friendly_support"].empathy >= 0.8

    def test_casual_tutor_low_formality(self):
        assert PRESETS["casual_tutor"].formality <= 0.4

    def test_analytical_assistant_low_empathy(self):
        assert PRESETS["analytical_assistant"].empathy <= 0.2

    def test_trait_docs_cover_all_traits(self):
        expected = {"formality", "empathy", "technical_depth", "verbosity",
                    "assertiveness", "humor", "politeness", "curiosity"}
        assert set(TRAIT_DOCS.keys()) == expected


# ===========================================================================
# SECTION 2: Controller Tests
# ===========================================================================

class TestLevelHelper:
    @pytest.mark.parametrize("val,expected", [
        (0.0, "low"), (0.1, "low"), (0.34, "low"),
        (0.35, "medium"), (0.5, "medium"), (0.64, "medium"),
        (0.65, "high"), (0.9, "high"), (1.0, "high"),
    ])
    def test_level_thresholds(self, val, expected):
        assert _level(val) == expected


class TestGeneratePromptConstraints:
    def test_output_is_string(self):
        p = Persona(name="t")
        result = generate_prompt_constraints(p)
        assert isinstance(result, str)

    def test_output_contains_persona_name(self):
        p = Persona(name="my_persona")
        result = generate_prompt_constraints(p)
        assert "my_persona" in result

    def test_all_trait_labels_present(self):
        p = Persona(name="t")
        result = generate_prompt_constraints(p)
        for label in ["Formality", "Empathy", "Technical Depth", "Verbosity",
                      "Assertiveness", "Humor", "Politeness", "Curiosity", "Tone"]:
            assert label in result

    def test_high_formality_yields_formal_instruction(self):
        p = Persona(name="t", formality=0.9)
        result = generate_prompt_constraints(p)
        assert "formal" in result.lower()

    def test_low_formality_yields_casual_instruction(self):
        p = Persona(name="t", formality=0.1)
        result = generate_prompt_constraints(p)
        assert "casual" in result.lower()

    def test_high_empathy_mentions_emotions(self):
        p = Persona(name="t", empathy=0.95)
        result = generate_prompt_constraints(p)
        assert "emotion" in result.lower() or "feelings" in result.lower()

    def test_low_empathy_suppresses_emotion(self):
        p = Persona(name="t", empathy=0.0)
        result = generate_prompt_constraints(p)
        assert "do not comment" in result.lower() or "purely on information" in result.lower()

    def test_high_verbosity_mentions_detail(self):
        p = Persona(name="t", verbosity=0.9)
        result = generate_prompt_constraints(p)
        assert "detail" in result.lower() or "thorough" in result.lower()

    def test_low_verbosity_mentions_concise(self):
        p = Persona(name="t", verbosity=0.0)
        result = generate_prompt_constraints(p)
        assert "concise" in result.lower() or "terse" in result.lower() or "sentences" in result.lower()

    def test_warm_tone_reflected_in_output(self):
        p = Persona(name="t", tone="warm")
        result = generate_prompt_constraints(p)
        assert "warm" in result.lower()

    def test_analytical_tone_reflected_in_output(self):
        p = Persona(name="t", tone="analytical")
        result = generate_prompt_constraints(p)
        assert "analytical" in result.lower()

    def test_different_personas_yield_different_prompts(self):
        p1 = PRESETS["formal_expert"]
        p2 = PRESETS["friendly_support"]
        assert generate_prompt_constraints(p1) != generate_prompt_constraints(p2)

    def test_prompt_ends_with_consistency_reminder(self):
        p = Persona(name="t")
        result = generate_prompt_constraints(p)
        assert "consistently" in result.lower()


class TestGenerateCorrectiveConstraint:
    def test_returns_string(self):
        result = generate_corrective_constraint("formality", "high")
        assert isinstance(result, str)

    def test_contains_correction_keyword(self):
        result = generate_corrective_constraint("empathy", "low")
        assert "CORRECTION" in result

    def test_formality_high_correction_mentions_formal(self):
        result = generate_corrective_constraint("formality", "high")
        assert "formal" in result.lower()

    def test_verbosity_low_correction_mentions_concise(self):
        result = generate_corrective_constraint("verbosity", "low")
        assert "concise" in result.lower() or "long" in result.lower()

    def test_unknown_trait_returns_fallback(self):
        result = generate_corrective_constraint("unknown_trait", "medium")
        assert "CORRECTION" in result
        assert "unknown_trait" in result

    def test_unknown_level_returns_fallback(self):
        result = generate_corrective_constraint("formality", "extreme")
        assert "CORRECTION" in result

    @pytest.mark.parametrize("trait", ["formality", "empathy", "verbosity", "technical_depth", "politeness"])
    def test_all_supported_traits_return_correction(self, trait):
        for level in ["low", "medium", "high"]:
            result = generate_corrective_constraint(trait, level)
            assert "CORRECTION" in result
