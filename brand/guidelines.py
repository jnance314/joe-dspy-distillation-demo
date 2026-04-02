"""
Liquid Death brand voice guidelines.

Edit this file to change the brand rules. The guidelines are injected into
the DSPy signature as context so the model knows what "on-brand" means.
"""

BRAND_NAME = "Liquid Death"

VOICE_DESCRIPTION = (
    "Liquid Death is a canned water brand that markets itself like a punk rock "
    "energy drink. The voice is irreverent, self-aware, and aggressively casual. "
    "It mocks corporate marketing while being corporate marketing. Think: your "
    "funniest friend who happens to sell water and knows the whole thing is absurd. "
    "Copy should feel like it was written by someone who hates ads but is weirdly "
    "good at writing them."
)

RULES = [
    "Use casual, conversational tone — like texting a friend who gets your humor",
    "Never use corporate jargon or formal marketing speak",
    "Keep sentences short and punchy — no walls of text",
    "Be self-aware — we know we're selling water and that's inherently funny",
    "Anti-establishment energy — we're the opposite of clean wellness brands",
    "Contractions always — 'we're' not 'we are', 'don't' not 'do not'",
    "Talk TO people, not AT them — second person ('you') over third person",
    "Humor should be dry and deadpan, not try-hard or slapstick",
    "Never beg for attention — no 'Buy now!' or 'Don't miss out!'",
    "Swearing is fine in moderation but never forced or edgy for the sake of it",
]

BANNED_PHRASES = [
    "premium quality",
    "best-in-class",
    "innovative solution",
    "we're committed to",
    "don't hesitate to",
    "leverage",
    "synergy",
    "world-class",
    "cutting-edge",
    "hydration solution",
    "please don't hesitate",
    "valued customer",
    "at your earliest convenience",
    "industry-leading",
    "holistic approach",
    "empower",
    "optimize your wellness",
    "curated experience",
    "seamless integration",
    "thought leader",
    "paradigm shift",
    "move the needle",
    "circle back",
    "low-hanging fruit",
    "revolutionary",
    "game-changing",
    "best practices",
    "core competency",
    "actionable insights",
    "robust platform",
]

PREFERRED_ALTERNATIVES = {
    "purchase": "grab",
    "consume": "drink",
    "beverage": "tallboy",
    "customer": "human",
    "utilize": "use",
    "facilitate": "help",
    "implement": "do",
    "regarding": "about",
    "in order to": "to",
    "at this time": "right now",
    "going forward": "from now on",
    "reach out": "hit us up",
    "provide feedback": "tell us what you think",
    "experience": "thing",
    "solution": "thing that works",
}

TONE_KEYWORDS = [
    "irreverent",
    "punk",
    "blunt",
    "deadpan",
    "casual",
    "anti-corporate",
    "self-aware",
    "absurdist",
]

# Structural rules used by the deterministic metric
MAX_SENTENCE_LENGTH_WORDS = 20
PASSIVE_VOICE_ALLOWED = False


def format_guidelines_prompt() -> str:
    """Format the guidelines into a single string for the DSPy signature input."""
    sections = [
        f"Brand: {BRAND_NAME}",
        f"\nVoice: {VOICE_DESCRIPTION}",
        "\nRules:",
        *[f"  - {r}" for r in RULES],
        "\nBanned phrases (never use these):",
        *[f"  - \"{p}\"" for p in BANNED_PHRASES],
        "\nPreferred alternatives:",
        *[f"  - Instead of \"{k}\", say \"{v}\"" for k, v in PREFERRED_ALTERNATIVES.items()],
        f"\nTone keywords: {', '.join(TONE_KEYWORDS)}",
        f"\nMax sentence length: {MAX_SENTENCE_LENGTH_WORDS} words",
        f"\nPassive voice allowed: {'Yes' if PASSIVE_VOICE_ALLOWED else 'No'}",
    ]
    return "\n".join(sections)
