"""
demo.py
-------
Quick manual test to see what the controller produces.
Run with: python demo.py
"""

from persona_schema import Persona, PRESETS
from persona_controller import generate_prompt_constraints, generate_corrective_constraint

DIVIDER = "=" * 70


def show_persona_prompt(persona: Persona):
    print(f"\n{DIVIDER}")
    print(f"  PERSONA: {persona.name.upper()}")
    print(DIVIDER)
    print(generate_prompt_constraints(persona))


def show_corrective(trait: str, level: str):
    print(f"\n  >> Corrective signal for '{trait}' drifted to '{level}':")
    print(f"     {generate_corrective_constraint(trait, level)}")


# --- Show all 4 presets ---
for persona in PRESETS.values():
    show_persona_prompt(persona)

# --- Show a custom persona ---
print(f"\n{DIVIDER}")
print("  PERSONA: CUSTOM (high empathy + low formality + high curiosity)")
print(DIVIDER)
custom = Persona(
    name="custom_bot",
    formality=0.1,
    empathy=0.95,
    technical_depth=0.3,
    verbosity=0.5,
    assertiveness=0.2,
    humor=0.6,
    politeness=0.9,
    curiosity=0.9,
    tone="warm",
)
print(generate_prompt_constraints(custom))

# --- Show corrective signals ---
print(f"\n{DIVIDER}")
print("  CORRECTIVE SIGNALS (simulating drift detection by Member 3)")
print(DIVIDER)
show_corrective("formality", "high")
show_corrective("empathy", "low")
show_corrective("verbosity", "low")
show_corrective("technical_depth", "high")
