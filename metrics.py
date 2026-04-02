"""
Deterministic evaluation metrics for the brand voice compliance checker.

All scoring is rule-based — no LLM-as-judge. This is intentional:
the demo argues that DSPy's value is in the optimized prompt, not in
a judge loop. If you could just loop a judge, you wouldn't need DSPy.

Composite metric = 0.4 * compliance + 0.3 * phrase_f1 + 0.3 * suggestion_quality
"""

import re

from brand.guidelines import (
    BANNED_PHRASES,
    MAX_SENTENCE_LENGTH_WORDS,
    PASSIVE_VOICE_ALLOWED,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_phrases(raw: str) -> set[str]:
    """Parse a comma-separated string of phrases into a normalized set."""
    if not raw or not raw.strip():
        return set()
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _has_passive_voice(text: str) -> bool:
    """Simple regex check for common passive voice patterns."""
    passive_pattern = r"\b(is|are|was|were|be|been|being)\s+\w+ed\b"
    return bool(re.search(passive_pattern, text, re.IGNORECASE))


def _avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    if not text or not text.strip():
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def _contains_banned_phrase(text: str) -> list[str]:
    """Return list of banned phrases found in text."""
    text_lower = text.lower()
    return [bp for bp in BANNED_PHRASES if bp.lower() in text_lower]


# ── Sub-metrics ──────────────────────────────────────────────────────────────

def compliance_accuracy(example, pred, trace=None) -> float:
    """1.0 if the compliant/non-compliant label matches, 0.0 otherwise."""
    expected = str(example.compliant).strip().lower()
    predicted = str(pred.compliant).strip().lower()
    return 1.0 if expected == predicted else 0.0


def phrase_detection_f1(example, pred, trace=None) -> float:
    """F1 score on flagged phrases using set overlap.

    Returns 1.0 if both expected and predicted are empty (true negative).
    """
    expected = _parse_phrases(example.flagged_phrases)
    predicted = _parse_phrases(pred.flagged_phrases)

    # Both empty = correct true negative
    if not expected and not predicted:
        return 1.0
    # One empty, other not = complete miss
    if not expected or not predicted:
        return 0.0

    # Fuzzy match: a predicted phrase counts if it's a substring of (or contains)
    # an expected phrase, to handle minor wording differences
    true_positives = 0
    matched_expected = set()
    for p in predicted:
        for e in expected:
            if p in e or e in p:
                true_positives += 1
                matched_expected.add(e)
                break

    precision = true_positives / len(predicted) if predicted else 0.0
    recall = len(matched_expected) / len(expected) if expected else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def suggestion_quality(example, pred, trace=None) -> float:
    """Rule-based quality checks on the suggested replacement.

    Checks:
      1. No banned phrases in the suggestion
      2. No passive voice (unless allowed by guidelines)
      3. Average sentence length under the max

    Returns 1.0 if compliant (no suggestion needed).
    Returns weighted score [0.0, 1.0] based on how many checks pass.
    """
    # If the copy is compliant, no suggestion is needed — full marks
    if str(example.compliant).strip().lower() == "true":
        return 1.0

    suggestion = str(pred.suggestion).strip() if hasattr(pred, "suggestion") else ""

    # No suggestion provided for non-compliant copy = 0
    if not suggestion:
        return 0.0

    checks_passed = 0
    total_checks = 3

    # Check 1: No banned phrases
    if not _contains_banned_phrase(suggestion):
        checks_passed += 1

    # Check 2: No passive voice
    if PASSIVE_VOICE_ALLOWED or not _has_passive_voice(suggestion):
        checks_passed += 1

    # Check 3: Sentence length
    if _avg_sentence_length(suggestion) <= MAX_SENTENCE_LENGTH_WORDS:
        checks_passed += 1

    return checks_passed / total_checks


# ── Composite metric (this is what MIPROv2 optimizes against) ────────────────

def composite_metric(example, pred, trace=None) -> float:
    """Weighted combination of all sub-metrics.

    Weights:
      - 0.4: compliance accuracy (did it get the label right?)
      - 0.3: phrase detection F1 (did it find the right problems?)
      - 0.3: suggestion quality (is the fix actually on-brand?)
    """
    ca = compliance_accuracy(example, pred, trace)
    pf = phrase_detection_f1(example, pred, trace)
    sq = suggestion_quality(example, pred, trace)
    return 0.4 * ca + 0.3 * pf + 0.3 * sq
