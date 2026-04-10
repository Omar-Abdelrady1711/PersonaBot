"""
alignment_scorer.py
-------------------
Measures how well a bot response matches the active persona.
Uses Claude (Anthropic API) as an independent LLM judge.

Deliberately uses a different model family from the response generator
(Azure OpenAI) to avoid self-evaluation bias.

Member 3 deliverable.

Install: pip install anthropic
"""

import json
import os
from dotenv import load_dotenv
import anthropic
from persona_schema import Persona

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Haiku is the cheapest model — fast and accurate enough for scoring
MODEL = "claude-haiku-4-5-20251001"

SCORING_PROMPT = """You are an expert evaluator of conversational style.
Your task is to measure how well a chatbot response matches a target persona.
Each persona defines a target level (0.0–1.0) for each trait.

IMPORTANT:
You are NOT scoring how much of a trait is present.
You are scoring how CLOSE the response is to the target value.
This means:
- If the target is LOW and the response is LOW → HIGH score
- If the target is HIGH and the response is HIGH → HIGH score
- If they differ significantly → LOW score

Use this rubric:
- 0.9–1.0 → Excellent match (very close to target)
- 0.7–0.8 → Good match (minor deviation)
- 0.4–0.6 → Partial match
- 0.1–0.3 → Poor match
- 0.0     → Opposite of target

Target persona:
- formality:       {formality}
- empathy:         {empathy}
- technical_depth: {technical_depth}
- verbosity:       {verbosity}
- assertiveness:   {assertiveness}
- humor:           {humor}
- politeness:      {politeness}
- curiosity:       {curiosity}

Response:
\"\"\"{response}\"\"\"

Return ONLY valid JSON with these keys. No explanation, no markdown, raw JSON only:
{{
  "formality": float,
  "empathy": float,
  "technical_depth": float,
  "verbosity": float,
  "assertiveness": float,
  "humor": float,
  "politeness": float,
  "curiosity": float
}}"""


def score_alignment(bot_response: str, persona: Persona) -> dict:
    """
    Uses Claude Haiku as an independent LLM judge to score how well
    a bot response matches the active persona.

    Args:
        bot_response: The bot's response string.
        persona:      The active Persona object (Member 1's dataclass).

    Returns:
        Dict with per-trait alignment scores (0.0-1.0) and overall score.
    """
    traits = ["formality", "empathy", "technical_depth", "verbosity",
              "assertiveness", "humor", "politeness", "curiosity"]

    if not bot_response or not bot_response.strip():
        return {t: 0.0 for t in traits} | {"overall": 0.0}

    prompt = SCORING_PROMPT.format(
        formality=persona.formality,
        empathy=persona.empathy,
        technical_depth=persona.technical_depth,
        verbosity=persona.verbosity,
        assertiveness=persona.assertiveness,
        humor=persona.humor,
        politeness=persona.politeness,
        curiosity=persona.curiosity,
        response=bot_response.replace('"""', "'''"),
    )

    response = _client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores = json.loads(raw)

    # Clamp all values to [0.0, 1.0]
    for t in traits:
        scores[t] = round(max(0.0, min(1.0, float(scores.get(t, 0.5)))), 4)

    scores["overall"] = round(sum(scores[t] for t in traits) / len(traits), 4)
    return scores


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from persona_schema import PRESETS

    test_cases = [
        (
            "formal_expert",
            "The transformer architecture employs multi-head self-attention mechanisms "
            "to compute contextual representations. This enables parallelization and "
            "captures long-range dependencies more efficiently than recurrent models."
        ),
        (
            "friendly_support",
            "Hey, I totally get how you're feeling — that sounds really overwhelming! "
            "You're not alone in this. What part is stressing you out the most? "
            "I'd love to help figure it out together 😊"
        ),
        (
            "casual_tutor",
            "Okay so basically, gradient descent is just the model 'learning' by "
            "nudging itself in the direction that makes it less wrong each time. "
            "Think of it like hiking downhill blindfolded — what part feels fuzzy?"
        ),
        (
            # Mismatched — casual response vs formal persona
            "formal_expert",
            "hey so basically transformers just look at all the words at once lol, "
            "pretty cool right? way better than the old stuff"
        ),
    ]

    labels = [
        "formal_expert    (matched)",
        "friendly_support (matched)",
        "casual_tutor     (matched)",
        "formal_expert    (MISMATCHED — casual response)",
    ]

    for label, (preset_name, response) in zip(labels, test_cases):
        persona = PRESETS[preset_name]
        scores = score_alignment(response, persona)
        print(f"\n--- {label} ---")
        print(f"Response: {response[:70]}...")
        print(f"Overall:  {scores['overall']:.3f}")
        for t in ["formality", "empathy", "technical_depth", "verbosity",
                  "assertiveness", "humor", "politeness", "curiosity"]:
            print(f"  {t:<18} {scores[t]:.3f}")