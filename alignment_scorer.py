"""
alignment_scorer.py
-------------------
Measures how well a bot response matches the active persona.
Uses Claude Haiku as an independent LLM judge.

Deliberately uses a different model family from the response generator
(Azure OpenAI) to avoid self-evaluation bias — a documented issue in
LLM evaluation (Ye et al., 2024).

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
MODEL = "claude-haiku-4-5-20251001"

SCORING_PROMPT = """You are an expert evaluator of conversational style.
Your task is to measure how well a chatbot response matches a target persona.
Each persona defines a target level (0.0–1.0) for each trait.

CORE RULE:
You are scoring CLOSENESS to the target value — not how much of a trait is present.
- If the target is LOW and the response is LOW → HIGH score
- If the target is HIGH and the response is HIGH → HIGH score
- If they differ significantly → LOW score

CRITICAL RULE — EXTREME TRAITS:
Traits with extreme target values (below 0.2 or above 0.8) are the most
important to match. Failing on an extreme trait must result in a low score
for that trait regardless of how well other traits match.

Specific guidance:
- empathy target {empathy}: if this is LOW (< 0.3), ANY emotional language,
  sympathy phrases like "I'm sorry", "I understand how you feel", personal
  encouragement, or emotional support must score VERY LOW (0.0–0.2) on empathy.
  A cold, factual response to an emotional message is CORRECT for a low-empathy persona.
- formality target {formality}: if this is HIGH (> 0.7), casual language,
  contractions, colloquialisms, or informal phrasing must score VERY LOW on formality.
- technical_depth target {technical_depth}: if this is HIGH (> 0.7), simple
  analogies, plain language, or non-technical explanations must score VERY LOW.
- technical_depth target {technical_depth}: if this is LOW (< 0.3), heavy
  jargon or domain-specific vocabulary must score VERY LOW.
- verbosity target {verbosity}: if this is LOW (< 0.3), long detailed responses
  must score VERY LOW. If HIGH (> 0.7), very short responses must score VERY LOW.

Scoring rubric per trait:
- 0.9–1.0 → Excellent match (very close to target)
- 0.7–0.8 → Good match (minor deviation)
- 0.4–0.6 → Partial match (noticeable deviation)
- 0.1–0.3 → Poor match (significant deviation)
- 0.0     → Opposite of target

Target persona values:
- formality:       {formality}
- empathy:         {empathy}
- technical_depth: {technical_depth}
- verbosity:       {verbosity}
- assertiveness:   {assertiveness}
- humor:           {humor}
- politeness:      {politeness}
- curiosity:       {curiosity}

Response to evaluate:
\"\"\"{response}\"\"\"

Return ONLY valid JSON with exactly these keys. No explanation, no markdown, raw JSON only:
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


if __name__ == "__main__":
    from persona_schema import PRESETS

    print("=== Testing updated scorer with extreme trait guidance ===\n")

    test_cases = [
        ("analytical_assistant",
         "I'm so sorry to hear you're frustrated! That's completely normal. "
         "Let me explain this in a simple way using a real-life example. "
         "Think of it like memorizing for a test — it's okay to find this hard!",
         "BAD — warm emotional response vs low-empathy persona"),

        ("analytical_assistant",
         "Overfitting occurs when a model captures noise in training data, "
         "resulting in high variance and poor generalization. Regularization "
         "techniques such as L2 penalty or dropout mitigate this.",
         "GOOD — technical cold response matches analytical_assistant"),

        ("friendly_support",
         "I completely understand how overwhelming this feels. You're not alone "
         "and it's okay to feel this way. What part is hardest for you right now?",
         "GOOD — warm empathetic response matches friendly_support"),

        ("friendly_support",
         "Overfitting is a statistical phenomenon characterized by excessive "
         "model complexity relative to the available training data.",
         "BAD — cold technical response vs high-empathy persona"),
    ]

    for preset_name, response, label in test_cases:
        persona = PRESETS[preset_name]
        scores = score_alignment(response, persona)
        print(f"[{label}]")
        print(f"Persona: {preset_name}")
        print(f"Response: {response[:70]}...")
        print(f"Overall: {scores['overall']:.3f} | "
              f"Empathy: {scores['empathy']:.3f} | "
              f"Formality: {scores['formality']:.3f} | "
              f"Tech: {scores['technical_depth']:.3f}")
        print()