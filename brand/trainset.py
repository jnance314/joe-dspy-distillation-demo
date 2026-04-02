"""
Training set for the brand voice compliance checker.

~30 labeled examples. Edit these to change what the optimizer learns from.

Each example has:
  - marketing_copy: a marketing copy snippet (1-3 sentences)
  - compliant: "true" or "false"
  - flagged_phrases: comma-separated problematic phrases (empty string if compliant)
  - suggestion: corrected on-brand version (empty string if already compliant)
"""

import dspy


def _ex(marketing_copy: str, compliant: str, flagged_phrases: str, suggestion: str) -> dspy.Example:
    return dspy.Example(
        marketing_copy=marketing_copy,
        compliant=compliant,
        flagged_phrases=flagged_phrases,
        suggestion=suggestion,
    ).with_inputs("marketing_copy")


trainset = [
    # ── COMPLIANT examples (on-brand) ────────────────────────────────────

    _ex(
        "Don't be the person who brings a plastic bottle to a party. Grab a tallboy. Your recycling bin will thank you.",
        "true", "", "",
    ),
    _ex(
        "We sell water. In a can. You're welcome.",
        "true", "", "",
    ),
    _ex(
        "Other water brands want you to feel zen. We want you to feel like you just crowd-surfed at a show.",
        "true", "", "",
    ),
    _ex(
        "Your tap water has been through some stuff. Ours comes from the Alps. Same price, way less trauma.",
        "true", "", "",
    ),
    _ex(
        "Yeah it's just water. But it's water that looks sick on your desk.",
        "true", "", "",
    ),
    _ex(
        "Hydration doesn't have to be boring. It also doesn't have to be interesting. It just has to be in a can.",
        "true", "", "",
    ),
    _ex(
        "We put water in a tallboy because bottles are for people who trust airport wifi.",
        "true", "", "",
    ),
    _ex(
        "If you're reading this, you're probably thirsty. Or bored. Either way, we can help with one of those.",
        "true", "", "",
    ),
    _ex(
        "Our whole thing is selling water to people who think water brands are dumb. And honestly, same.",
        "true", "", "",
    ),
    _ex(
        "You don't need another wellness brand telling you to hydrate. But you do need a tallboy. Trust us.",
        "true", "", "",
    ),
    _ex(
        "Liquid Death: for people who want to drink water but make it look like they're drinking something cooler.",
        "true", "", "",
    ),
    _ex(
        "We're not a lifestyle brand. We're water in a can. Stop overthinking it.",
        "true", "", "",
    ),

    # ── NON-COMPLIANT examples (off-brand) ───────────────────────────────

    # Corporate jargon
    _ex(
        "Liquid Death is committed to providing a premium quality hydration solution for today's discerning consumer.",
        "false",
        "committed to providing, premium quality, hydration solution",
        "We made water that doesn't suck. You're the type of person who gets that.",
    ),
    _ex(
        "Our innovative solution leverages cutting-edge filtration to deliver a best-in-class beverage experience.",
        "false",
        "innovative solution, leverages, cutting-edge, best-in-class, beverage experience",
        "We filter our water really well. It tastes good. That's the whole pitch.",
    ),
    _ex(
        "We would like to circle back on our holistic approach to sustainable hydration and move the needle on industry best practices.",
        "false",
        "circle back, holistic approach, move the needle, best practices",
        "We're trying to make water less wasteful. Cans recycle better than plastic. Done.",
    ),
    _ex(
        "At Liquid Death, we empower our valued customers to optimize their wellness journey through our curated experience.",
        "false",
        "empower, valued customers, optimize, wellness journey, curated experience",
        "Drink water. Feel fine. That's the whole journey.",
    ),

    # Too formal / stiff
    _ex(
        "We invite you to explore our extensive range of premium hydration products at your earliest convenience.",
        "false",
        "at your earliest convenience, premium, extensive range",
        "We've got water. Still or sparkling. Grab one whenever.",
    ),
    _ex(
        "Please do not hesitate to reach out to our customer support team regarding any inquiries you may have.",
        "false",
        "please do not hesitate, reach out, regarding, inquiries",
        "Got a question? Hit us up. We actually read these.",
    ),
    _ex(
        "Liquid Death is pleased to announce the launch of our revolutionary new sparkling water product line.",
        "false",
        "pleased to announce, revolutionary, product line",
        "New sparkling water just dropped. It's bubbly and it's in a can. You know the drill.",
    ),

    # Passive voice
    _ex(
        "Our cans are crafted with care and designed to be enjoyed by health-conscious individuals.",
        "false",
        "are crafted with care, designed to be enjoyed, health-conscious individuals",
        "We make cans of water. You drink them. Nobody needs to be 'health-conscious' about it.",
    ),
    _ex(
        "A portion of proceeds is donated to environmental initiatives that are supported by our organization.",
        "false",
        "is donated, are supported by our organization",
        "We give some of the money to help the planet. Because, you know, we live here too.",
    ),

    # Pushy sales / desperate
    _ex(
        "Buy now and don't miss out on this incredible limited-time offer! Act fast before supplies run out!",
        "false",
        "Buy now, don't miss out, incredible, limited-time offer, Act fast",
        "We made some stuff. There's not a ton of it. If you want some, cool.",
    ),
    _ex(
        "Subscribe today and save 20%! This game-changing deal won't last forever!",
        "false",
        "Subscribe today, game-changing, won't last forever",
        "You can subscribe and save 20%. Or don't. We're not your mom.",
    ),

    # Too long / wordy
    _ex(
        "At Liquid Death, we believe that every single person on this planet deserves access to clean, refreshing, mountain-sourced water that not only quenches their thirst but also contributes to a more sustainable and environmentally responsible future for generations to come.",
        "false",
        "we believe that every single person on this planet deserves access to, not only quenches their thirst but also contributes to, for generations to come",
        "Clean water from the mountains. In a can that actually gets recycled. The end.",
    ),

    # Wellness brand energy (wrong vibe)
    _ex(
        "Nourish your body and nurture your soul with the pure, life-giving essence of mountain spring water.",
        "false",
        "nourish your body, nurture your soul, life-giving essence",
        "It's water from a mountain. Drink it. Your body will figure out the rest.",
    ),
    _ex(
        "Begin your mindful hydration ritual with Liquid Death — because self-care starts from within.",
        "false",
        "mindful hydration ritual, self-care starts from within",
        "Drink some water. It's not a ritual. It's just water.",
    ),

    # Third person / talking AT people
    _ex(
        "Consumers who choose Liquid Death are making a responsible choice for the environment and their personal health.",
        "false",
        "consumers, making a responsible choice",
        "You grabbed a can of water. Congrats, you recycled something today. Don't let it go to your head.",
    ),
    _ex(
        "The Liquid Death brand represents a paradigm shift in how the beverage industry approaches sustainability.",
        "false",
        "paradigm shift, beverage industry, approaches sustainability",
        "We put water in cans instead of plastic bottles. It's not rocket science but somehow nobody else did it.",
    ),
]
