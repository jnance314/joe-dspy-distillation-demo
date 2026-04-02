"""
Validation set for the brand voice compliance checker.

~15 labeled examples, held out from training. Never seen during optimization.
Same structure as trainset.py — edit independently.
"""

import dspy


def _ex(marketing_copy: str, compliant: str, flagged_phrases: str, suggestion: str) -> dspy.Example:
    return dspy.Example(
        marketing_copy=marketing_copy,
        compliant=compliant,
        flagged_phrases=flagged_phrases,
        suggestion=suggestion,
    ).with_inputs("marketing_copy")


valset = [
    # ── COMPLIANT ────────────────────────────────────────────────────────

    _ex(
        "You could drink tap water. But tap water never headlined a music festival.",
        "true", "", "",
    ),
    _ex(
        "Sparkling water for people who think sparkling water is for fancy people. Plot twist: you're fancy now.",
        "true", "", "",
    ),
    _ex(
        "It's water. From the Alps. In a can. We're not going to overcomplicate this.",
        "true", "", "",
    ),
    _ex(
        "Our merch is dumb and we love it. Tallboy koozies, skull hats, the whole thing.",
        "true", "", "",
    ),
    _ex(
        "Hey, we get it. Drinking water is boring. But have you tried drinking water that looks like a beer?",
        "true", "", "",
    ),
    _ex(
        "Subscribe to Liquid Death and we'll mail you water every month. Like a magazine but heavier and wetter.",
        "true", "", "",
    ),

    # ── NON-COMPLIANT ────────────────────────────────────────────────────

    _ex(
        "Liquid Death's robust platform delivers an industry-leading hydration experience powered by actionable insights from consumer research.",
        "false",
        "robust platform, industry-leading, hydration experience, actionable insights",
        "We sell water. People like it. That's all the research we need.",
    ),
    _ex(
        "We are dedicated to facilitating a seamless integration of sustainable practices into every facet of our operations.",
        "false",
        "facilitating, seamless integration, facet of our operations",
        "We try not to trash the planet. Cans over plastic. Pretty simple.",
    ),
    _ex(
        "Explore our thoughtfully curated collection of artisanal mountain water, crafted for the modern wellness enthusiast.",
        "false",
        "thoughtfully curated collection, artisanal, crafted for, wellness enthusiast",
        "Mountain water in a can. It's not artisanal. It's just good.",
    ),
    _ex(
        "Each Liquid Death product is meticulously engineered to provide maximum refreshment while maintaining our unwavering commitment to quality.",
        "false",
        "meticulously engineered, maximum refreshment, unwavering commitment to quality",
        "We made the water taste good. That's basically our whole job.",
    ),
    _ex(
        "Don't miss this exclusive opportunity! Order your limited-edition Liquid Death bundle now before it's gone forever!",
        "false",
        "exclusive opportunity, limited-edition, now before it's gone forever",
        "We made some limited bundles. They'll sell out eventually. No rush. Okay maybe a little rush.",
    ),
    _ex(
        "The purest mountain spring water is carefully sourced and bottled to ensure optimal freshness is delivered to your doorstep.",
        "false",
        "is carefully sourced, is delivered, optimal freshness",
        "We grab water from the mountains, stick it in cans, and ship it to you. Fresh? Yeah, it's water.",
    ),
    _ex(
        "Liquid Death invites all health-conscious individuals to embark on a transformative hydration journey that redefines personal wellness.",
        "false",
        "health-conscious individuals, transformative hydration journey, redefines personal wellness",
        "Drink water. Feel okay about it. That's the whole journey. You're done.",
    ),
    _ex(
        "Our organization leverages strategic partnerships with environmental thought leaders to drive meaningful change in the beverage industry.",
        "false",
        "leverages, strategic partnerships, thought leaders, meaningful change, beverage industry",
        "We work with people who care about the environment. Together we do stuff that matters. The end.",
    ),
    _ex(
        "Liquid Death's core competency lies in disrupting the traditional beverage paradigm through innovative, consumer-centric solutions.",
        "false",
        "core competency, paradigm, innovative, consumer-centric solutions",
        "We sell canned water. Nobody asked us to 'disrupt' anything. We just thought plastic bottles were dumb.",
    ),
]
