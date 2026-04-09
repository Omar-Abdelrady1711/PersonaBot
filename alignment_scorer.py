"""
alignment_scorer.py
-------------------
Measures how well a bot response matches the active persona.
Uses linguistic proxy metrics (textstat, vaderSentiment) to score each trait.

Member 3 deliverable — no dependency on Member 2 at runtime.
"""

import re
from persona_schema import Persona

# Optional imports — graceful fallback if not installed
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Low-level text feature extractors
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    sentences = re.split(r'[.!?]+', text.strip())
    return max(1, len([s for s in sentences if s.strip()]))


def _avg_sentence_length(text: str) -> float:
    return _word_count(text) / _sentence_count(text)


def _contraction_frequency(text: str) -> float:
    """Ratio of contractions to total words. High = casual."""
    contractions = re.findall(
        r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|"
        r"he's|she's|it's|we're|they're|don't|doesn't|didn't|"
        r"won't|wouldn't|can't|couldn't|shouldn't|isn't|aren't|"
        r"wasn't|weren't|haven't|hasn't|hadn't|that's|what's|let's)\b",
        text.lower()
    )
    words = _word_count(text)
    return len(contractions) / words if words > 0 else 0.0


def _second_person_ratio(text: str) -> float:
    """Ratio of 'you/your/you're' to total words. High = empathetic/engaging."""
    second_person = re.findall(r'\b(you|your|you\'re|you\'ve|you\'ll|you\'d)\b', text.lower())
    words = _word_count(text)
    return len(second_person) / words if words > 0 else 0.0


def _question_ratio(text: str) -> float:
    """Ratio of question sentences to total sentences."""
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    questions = [s for s in sentences if s.endswith('?') or text.count('?') > 0]
    question_marks = text.count('?')
    return min(1.0, question_marks / _sentence_count(text))


def _positive_sentiment(text: str) -> float:
    """Returns VADER positive sentiment score (0-1)."""
    if VADER_AVAILABLE:
        return _vader.polarity_scores(text)['pos']
    # Fallback: count positive words
    positive_words = ['great', 'wonderful', 'happy', 'glad', 'understand',
                      'help', 'excellent', 'good', 'sure', 'absolutely']
    words = text.lower().split()
    matches = sum(1 for w in words if w in positive_words)
    return min(1.0, matches / max(1, len(words)) * 10)


def _negative_sentiment(text: str) -> float:
    """Returns VADER negative sentiment score (0-1)."""
    if VADER_AVAILABLE:
        return _vader.polarity_scores(text)['neg']
    negative_words = ['bad', 'wrong', 'no', 'not', 'never', 'terrible', 'awful']
    words = text.lower().split()
    matches = sum(1 for w in words if w in negative_words)
    return min(1.0, matches / max(1, len(words)) * 10)


def _readability_score(text: str) -> float:
    """
    Flesch Reading Ease: 0-100. Higher = easier/simpler.
    We invert + normalize so high technical_depth → high score.
    """
    if TEXTSTAT_AVAILABLE:
        raw = textstat.flesch_reading_ease(text)
        # Clamp to [0, 100], invert (harder text = higher technical score)
        raw = max(0.0, min(100.0, raw))
        return round(1.0 - (raw / 100.0), 4)
    # Fallback: use avg word length as proxy
    words = text.split()
    if not words:
        return 0.5
    avg_word_len = sum(len(w) for w in words) / len(words)
    return min(1.0, max(0.0, (avg_word_len - 3) / 7))


def _humor_signal(text: str) -> float:
    """
    Rough humor proxy: presence of exclamation marks, emoji-like patterns,
    lol/haha, ellipses used for comic effect, and light punctuation.
    """
    signals = 0
    text_lower = text.lower()
    if re.search(r'\blol\b|\bhaha\b|\bhehe\b|\bxd\b', text_lower):
        signals += 3
    signals += min(3, text.count('!'))
    if re.search(r'😄|😂|🤣|😆|😜|😛|😝|🙃|😏', text):
        signals += 2
    if re.search(r'\.\.\.|…', text):
        signals += 1
    # Witty rhetorical questions
    if re.search(r'\?.*\?', text):
        signals += 1
    return min(1.0, signals / 8.0)


# ---------------------------------------------------------------------------
# Per-trait scorers
# ---------------------------------------------------------------------------

def _score_formality(response: str, persona: Persona) -> float:
    """
    High formality persona → expect long sentences, few contractions.
    Score = how close the response's formality proxy is to the persona's target.
    """
    avg_len = _avg_sentence_length(response)
    contraction_freq = _contraction_frequency(response)

    # Normalize avg sentence length: 5 words = low formality, 25+ = high
    formality_proxy = min(1.0, max(0.0, (avg_len - 5) / 20))
    # Contractions reduce formality
    formality_proxy = formality_proxy * 0.7 + (1.0 - min(1.0, contraction_freq * 20)) * 0.3

    return 1.0 - abs(formality_proxy - persona.formality)


def _score_empathy(response: str, persona: Persona) -> float:
    """
    High empathy persona → positive sentiment, second-person pronouns.
    """
    pos = _positive_sentiment(response)
    second = _second_person_ratio(response)

    empathy_proxy = pos * 0.5 + min(1.0, second * 10) * 0.5
    return 1.0 - abs(empathy_proxy - persona.empathy)


def _score_technical_depth(response: str, persona: Persona) -> float:
    """
    High technical_depth → harder to read (inverted Flesch score).
    """
    tech_proxy = _readability_score(response)
    return 1.0 - abs(tech_proxy - persona.technical_depth)


def _score_verbosity(response: str, persona: Persona) -> float:
    """
    High verbosity persona → longer responses. Normalize word count.
    Rough scale: 10 words = 0.0, 200+ words = 1.0.
    """
    wc = _word_count(response)
    verbosity_proxy = min(1.0, max(0.0, (wc - 10) / 190))
    return 1.0 - abs(verbosity_proxy - persona.verbosity)


def _score_politeness(response: str, persona: Persona) -> float:
    """
    High politeness → low negative sentiment, polite phrases.
    """
    neg = _negative_sentiment(response)
    polite_phrases = ['please', 'thank you', 'certainly', 'of course',
                      'happy to', 'glad to', 'appreciate', 'wonderful']
    text_lower = response.lower()
    polite_count = sum(1 for p in polite_phrases if p in text_lower)

    politeness_proxy = (1.0 - min(1.0, neg * 5)) * 0.6 + min(1.0, polite_count / 3) * 0.4
    return 1.0 - abs(politeness_proxy - persona.politeness)


def _score_humor(response: str, persona: Persona) -> float:
    humor_proxy = _humor_signal(response)
    return 1.0 - abs(humor_proxy - persona.humor)


def _score_curiosity(response: str, persona: Persona) -> float:
    """
    High curiosity → asks follow-up questions.
    """
    question_ratio = _question_ratio(response)
    curiosity_proxy = min(1.0, question_ratio * 2)
    return 1.0 - abs(curiosity_proxy - persona.curiosity)


def _score_assertiveness(response: str, persona: Persona) -> float:
    """
    High assertiveness → few hedging phrases.
    """
    hedges = ['maybe', 'perhaps', 'might', 'could be', 'possibly',
              'i think', 'it seems', 'sort of', 'kind of', 'not sure']
    text_lower = response.lower()
    hedge_count = sum(1 for h in hedges if h in text_lower)

    # Fewer hedges = higher assertiveness proxy
    assertiveness_proxy = max(0.0, 1.0 - min(1.0, hedge_count / 4))
    return 1.0 - abs(assertiveness_proxy - persona.assertiveness)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

TRAIT_SCORERS = {
    "formality":       _score_formality,
    "empathy":         _score_empathy,
    "technical_depth": _score_technical_depth,
    "verbosity":       _score_verbosity,
    "politeness":      _score_politeness,
    "humor":           _score_humor,
    "curiosity":       _score_curiosity,
    "assertiveness":   _score_assertiveness,
}


def score_alignment(bot_response: str, persona: Persona) -> dict:
    """
    Measures how well a bot response matches the active persona.

    Args:
        bot_response: The bot's response string.
        persona:      The active Persona object (Member 1's dataclass).

    Returns:
        Dict with per-trait scores (0.0–1.0) and an overall score.
        {
            "formality": 0.82,
            "empathy": 0.91,
            ...
            "overall": 0.85
        }
    """
    if not bot_response or not bot_response.strip():
        return {trait: 0.0 for trait in TRAIT_SCORERS} | {"overall": 0.0}

    scores = {}
    for trait, scorer in TRAIT_SCORERS.items():
        raw = scorer(bot_response, persona)
        scores[trait] = round(max(0.0, min(1.0, raw)), 4)

    scores["overall"] = round(sum(scores.values()) / len(scores), 4)
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
    ]

    for preset_name, response in test_cases:
        persona = PRESETS[preset_name]
        scores = score_alignment(response, persona)
        print(f"\nPersona: {preset_name}")
        print(f"Response snippet: {response[:60]}...")
        print(f"Scores: {scores}")