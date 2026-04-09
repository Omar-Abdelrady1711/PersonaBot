"""
tone_detector.py
----------------
Detects user personality traits from their message via the Azure OpenAI API.
Returns a Persona object representing the inferred user tone.
"""

import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from persona_schema import Persona

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

TONE_DETECTION_PROMPT = """
You are a personality analysis assistant.
Analyze the user message below and estimate the writer's communication style.
Return ONLY a valid JSON object with exactly these keys and float values between 0.0 and 1.0.
Do not include any explanation, preamble, or markdown — raw JSON only.

{{
  "formality": <0.0 casual to 1.0 formal>,
  "empathy": <0.0 cold to 1.0 emotionally expressive>,
  "technical_depth": <0.0 plain language to 1.0 expert jargon>,
  "verbosity": <0.0 very short to 1.0 very detailed>,
  "assertiveness": <0.0 tentative to 1.0 confident/direct>,
  "humor": <0.0 serious to 1.0 playful/witty>,
  "politeness": <0.0 blunt to 1.0 very polite>,
  "curiosity": <0.0 closed to 1.0 inquisitive>
}}

User message: "{message}"
"""


def _infer_tone_label(traits: dict) -> str:
    """Maps detected trait scores to one of Member 1's valid tone labels."""
    if traits.get("empathy", 0) >= 0.65:
        return "empathetic"
    if traits.get("humor", 0) >= 0.65:
        return "playful"
    if traits.get("formality", 0) >= 0.65 and traits.get("technical_depth", 0) >= 0.65:
        return "authoritative"
    if traits.get("technical_depth", 0) >= 0.65:
        return "analytical"
    if traits.get("formality", 0) <= 0.35 and traits.get("politeness", 0) >= 0.65:
        return "warm"
    return "neutral"


def detect_tone(user_message: str) -> Persona:
    """
    Calls the Azure OpenAI API to analyze the user's message and infer personality traits.
    Returns a Persona object representing the detected user tone.
    """
    prompt = TONE_DETECTION_PROMPT.format(message=user_message.replace('"', "'"))

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a personality analysis assistant. Return only raw JSON."},
            {"role": "user", "content": prompt},
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    detected = json.loads(raw)

    # Clamp all values to [0.0, 1.0] just in case
    for key in detected:
        detected[key] = max(0.0, min(1.0, float(detected[key])))

    tone = _infer_tone_label(detected)

    return Persona(
        name="detected_user",
        formality=detected.get("formality", 0.5),
        empathy=detected.get("empathy", 0.5),
        technical_depth=detected.get("technical_depth", 0.5),
        verbosity=detected.get("verbosity", 0.5),
        assertiveness=detected.get("assertiveness", 0.5),
        humor=detected.get("humor", 0.2),
        politeness=detected.get("politeness", 0.7),
        curiosity=detected.get("curiosity", 0.4),
        tone=tone,
    )


if __name__ == "__main__":
    test_messages = [
        "hey i'm kinda lost on this topic lol, can you help?",
        "I would appreciate a thorough technical explanation of transformer architectures.",
        "I'm feeling really overwhelmed and don't know what to do anymore.",
    ]
    for msg in test_messages:
        print(f"\nUser: {msg}")
        p = detect_tone(msg)
        print(f"  tone={p.tone}  formality={p.formality:.2f}  empathy={p.empathy:.2f}  "
              f"tech={p.technical_depth:.2f}  verbosity={p.verbosity:.2f}")
