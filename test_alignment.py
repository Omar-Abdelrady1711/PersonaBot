"""
test_alignment.py
-----------------
Quick manual tests for alignment_scorer.py.

Run with:
    python test_alignment.py

Purpose:
- sanity-check whether high/low scores make intuitive sense
- compare good vs bad responses for the same persona
- help tune heuristics quickly before full integration
"""

from persona_schema import PRESETS, Persona
from alignment_scorer import score_alignment

DIVIDER = "=" * 90


def show_case(title: str, persona: Persona, response: str):
    scores = score_alignment(response, persona)

    print(f"\n{DIVIDER}")
    print(f"CASE: {title}")
    print(f"TARGET PERSONA: {persona.name}")
    print(DIVIDER)
    print("Response:")
    print(response)
    print("\nScores:")
    for trait, value in scores.items():
        print(f"  {trait:16s} {value:.4f}")

    worst_trait = min(
        [k for k in scores.keys() if k != "overall"],
        key=lambda k: scores[k]
    )
    best_trait = max(
        [k for k in scores.keys() if k != "overall"],
        key=lambda k: scores[k]
    )

    print(f"\nBest-matching trait : {best_trait} ({scores[best_trait]:.4f})")
    print(f"Worst-matching trait: {worst_trait} ({scores[worst_trait]:.4f})")
    print(f"Overall alignment   : {scores['overall']:.4f}")


def compare_cases(title: str, persona: Persona, good_response: str, bad_response: str):
    good_scores = score_alignment(good_response, persona)
    bad_scores = score_alignment(bad_response, persona)

    print(f"\n{DIVIDER}")
    print(f"COMPARISON: {title}")
    print(f"TARGET PERSONA: {persona.name}")
    print(DIVIDER)

    print("\nGOOD RESPONSE:")
    print(good_response)
    print(f"\nGOOD overall: {good_scores['overall']:.4f}")

    print("\nBAD RESPONSE:")
    print(bad_response)
    print(f"\nBAD overall:  {bad_scores['overall']:.4f}")

    delta = round(good_scores["overall"] - bad_scores["overall"], 4)
    print(f"\nDifference (good - bad): {delta:.4f}")

    print("\nPer-trait comparison:")
    trait_names = [k for k in good_scores.keys() if k != "overall"]
    for trait in [k for k in good_scores.keys() if k not in ("overall", "overall_match")]:
        print(
            f"  {trait:16s} "
            f"good={good_scores[trait]:.4f}   "
            f"bad={bad_scores[trait]:.4f}"
        )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Preset sanity checks: response intentionally matches persona
    # ------------------------------------------------------------------

    show_case(
        title="formal_expert - expected good match",
        persona=PRESETS["formal_expert"],
        response=(
            "The transformer architecture employs multi-head self-attention to compute "
            "context-sensitive token representations. This mechanism improves parallelization "
            "and captures long-range dependencies more effectively than recurrent approaches."
        )
    )

    show_case(
        title="friendly_support - expected good match",
        persona=PRESETS["friendly_support"],
        response=(
            "I’m really sorry you’re dealing with that. That sounds genuinely overwhelming, "
            "and you’re not alone in feeling this way. I’m here with you — what part feels "
            "hardest right now?"
        )
    )

    show_case(
        title="casual_tutor - expected good match",
        persona=PRESETS["casual_tutor"],
        response=(
            "Okay, so basically gradient descent is just the model getting a little less wrong "
            "each step. Think of it like walking downhill in fog and checking if each move helps. "
            "What part feels fuzzy?"
        )
    )

    show_case(
        title="analytical_assistant - expected good match",
        persona=PRESETS["analytical_assistant"],
        response=(
            "The key issue is variance reduction during estimation. A practical solution is to "
            "increase sample efficiency through stratified evaluation, then compare distributions "
            "under controlled assumptions before drawing conclusions."
        )
    )

    # ------------------------------------------------------------------
    # 2) Strong mismatch checks
    # ------------------------------------------------------------------

    show_case(
        title="formal_expert persona but casual playful response",
        persona=PRESETS["formal_expert"],
        response=(
            "Hey lol, it’s basically the model figuring out what words matter most, "
            "which is pretty cool if you think about it!"
        )
    )

    show_case(
        title="friendly_support persona but cold blunt response",
        persona=PRESETS["friendly_support"],
        response=(
            "You should focus on solving the problem directly. Emotional reactions are not useful."
        )
    )

    # ------------------------------------------------------------------
    # 3) Good vs bad comparisons for same persona
    # ------------------------------------------------------------------

    compare_cases(
        title="friendly_support: empathetic vs cold",
        persona=PRESETS["friendly_support"],
        good_response=(
            "I understand why this feels so heavy right now. That sounds really difficult, "
            "and you do not have to deal with it alone. What has been weighing on you the most?"
        ),
        bad_response=(
            "Do the task one step at a time. There is no reason to be overwhelmed."
        )
    )

    compare_cases(
        title="formal_expert: professional vs casual",
        persona=PRESETS["formal_expert"],
        good_response=(
            "This method is preferable because it preserves contextual dependencies while "
            "remaining computationally parallelizable. In practice, that makes it more scalable "
            "than recurrent alternatives."
        ),
        bad_response=(
            "Yeah, so it kind of just works better because it can look at everything at once."
        )
    )

    compare_cases(
        title="casual_tutor: curious vs non-curious",
        persona=PRESETS["casual_tutor"],
        good_response=(
            "Think of overfitting like memorizing answers instead of learning the lesson. "
            "The model looks smart on training data but struggles on new examples. "
            "What part of that feels confusing?"
        ),
        bad_response=(
            "Overfitting is when a model memorizes training data and fails to generalize."
        )
    )

    # ------------------------------------------------------------------
    # 4) Trait-specific probes
    # These help you debug one trait at a time.
    # ------------------------------------------------------------------

    show_case(
        title="Trait probe - very empathetic wording",
        persona=Persona(
            name="high_empathy_probe",
            empathy=0.95,
            formality=0.3,
            technical_depth=0.2,
            verbosity=0.5,
            assertiveness=0.3,
            humor=0.1,
            politeness=0.9,
            curiosity=0.7,
            tone="empathetic",
        ),
        response=(
            "I’m so sorry this has been so painful. That sounds exhausting, and I can see why "
            "you feel overwhelmed. You are not alone in this — what has been the hardest part?"
        )
    )

    show_case(
        title="Trait probe - highly technical wording",
        persona=Persona(
            name="high_technical_probe",
            empathy=0.1,
            formality=0.8,
            technical_depth=0.95,
            verbosity=0.7,
            assertiveness=0.8,
            humor=0.0,
            politeness=0.5,
            curiosity=0.2,
            tone="analytical",
        ),
        response=(
            "The optimization objective is non-convex, so convergence guarantees are limited. "
            "However, gradient-based approximation remains effective under suitable initialization "
            "and regularization assumptions."
        )
    )

    show_case(
        title="Trait probe - highly curious wording",
        persona=Persona(
            name="high_curiosity_probe",
            empathy=0.5,
            formality=0.4,
            technical_depth=0.3,
            verbosity=0.5,
            assertiveness=0.4,
            humor=0.2,
            politeness=0.8,
            curiosity=0.95,
            tone="warm",
        ),
        response=(
            "That makes sense. Can you tell me more about what happened first? "
            "What part felt most confusing to you? Would it help if we broke it down together?"
        )
    )

    # ------------------------------------------------------------------
    # 5) Edge cases
    # ------------------------------------------------------------------

    show_case(
        title="Edge case - empty response",
        persona=PRESETS["friendly_support"],
        response=""
    )

    show_case(
        title="Edge case - one short sentence",
        persona=PRESETS["friendly_support"],
        response="I can help."
    )