"""
Test set for the brand voice compliance checker.

~10 labeled examples, completely held out. Never seen during training OR
optimization (unlike valset, which MIPROv2 uses to score candidate prompts).

This is the only honest evaluation set -- no data leakage possible.
"""

import dspy


def _ex(marketing_copy: str, compliant: str, flagged_phrases: str, suggestion: str) -> dspy.Example:
    return dspy.Example(
        marketing_copy=marketing_copy,
        compliant=compliant,
        flagged_phrases=flagged_phrases,
        suggestion=suggestion,
    ).with_inputs("marketing_copy")


testset = [
    # -- COMPLIANT --

    _ex(
        "We didn't invent water. We just made it look cooler than your personality.",
        "true", "", "",
    ),
    _ex(
        "Sparkling or still? Doesn't matter. Both come in a can. Both judge your plastic bottle.",
        "true", "", "",
    ),
    _ex(
        "You could spend five bucks on a smoothie. Or you could grab a tallboy and still have money for bad decisions.",
        "true", "", "",
    ),
    _ex(
        "We're a water company that sponsors punk shows. Your move, Evian.",
        "true", "", "",
    ),

    # -- NON-COMPLIANT --

    _ex(
        "Liquid Death is proud to unveil our revolutionary new product line, designed to empower health-conscious consumers on their wellness journey.",
        "false",
        "proud to unveil, revolutionary, product line, empower, health-conscious consumers, wellness journey",
        "New stuff just dropped. It's water. You'll figure out the rest.",
    ),
    _ex(
        "Our team of hydration experts has crafted a seamless integration of flavor and function to deliver an industry-leading refreshment experience.",
        "false",
        "hydration experts, crafted, seamless integration, industry-leading, refreshment experience",
        "We made flavored water. It tastes good. That's the whole story.",
    ),
    _ex(
        "At Liquid Death, we leverage actionable insights from our valued customers to continuously optimize our robust platform for maximum impact.",
        "false",
        "leverage, actionable insights, valued customers, optimize, robust platform",
        "People tell us stuff. We listen. Then we make the water better.",
    ),
    _ex(
        "Don't miss this game-changing opportunity to experience our world-class hydration solution at your earliest convenience!",
        "false",
        "game-changing, experience, world-class, hydration solution, at your earliest convenience",
        "We sell water. Grab some whenever. No rush.",
    ),
    _ex(
        "A holistic approach is being taken by our organization to ensure that best practices are implemented across all core competencies.",
        "false",
        "holistic approach, is being taken, best practices, are implemented, core competencies",
        "We're trying to do things better. Across the board. That's it.",
    ),
    _ex(
        "Consumers who prioritize sustainable beverage choices will find that our premium quality products exceed expectations and move the needle on environmental responsibility.",
        "false",
        "consumers, sustainable beverage choices, premium quality, exceed expectations, move the needle",
        "You like the planet? Cool, us too. Our cans get recycled. Bottles don't. Easy choice.",
    ),
]
