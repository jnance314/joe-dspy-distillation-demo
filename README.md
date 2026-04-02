# DSPy Brand Voice Compliance Checker

Demonstrates [DSPy](https://dspy.ai/)'s automatic prompt optimization using MIPROv2.

## The problem

Writing prompts for brand voice compliance is painful. The rules are nuanced, the output is inconsistent, and if you want good results you're stuck paying for an expensive model. Switching models means re-doing all the prompt work.

## What this demo shows

1. A cheap model (Gemini 2.5 Flash Lite, $0.10/M tokens) tries to check brand compliance **zero-shot** — baseline accuracy
2. DSPy's MIPROv2 optimizer uses a smart teacher model (Gemini 2.5 Pro, $1.25/M tokens) to **automatically find the best prompt** — instructions + few-shot examples
3. The same cheap model, now with the optimized prompt, performs **significantly better**
4. The optimization runs once. After that, you deploy the cheap model with no teacher in the loop.

## Use case

**Liquid Death brand voice compliance** — given marketing copy and brand guidelines, the system:
- Labels copy as compliant or non-compliant
- Flags specific problematic phrases
- Suggests on-brand replacements

## Evaluation metrics (fully deterministic, no LLM-as-judge)

| Metric | What it measures |
|--------|-----------------|
| Compliance accuracy | Did it get the on-brand/off-brand label right? |
| Phrase detection F1 | Did it find the correct problematic phrases? |
| Suggestion quality | Does the replacement follow brand rules? (no banned phrases, no passive voice, short sentences) |
| **Composite** | Weighted combination: 40% compliance + 30% phrase F1 + 30% suggestion quality |

## Quick start

```bash
uv sync
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
uv run demo.py
```

## Files

| Path | Purpose |
|------|---------|
| `demo.py` | Main script — run this |
| `metrics.py` | Deterministic scoring functions |
| `brand/guidelines.py` | Brand voice rules, banned phrases, tone descriptors |
| `brand/trainset.py` | ~30 labeled training examples (edit to iterate) |
| `brand/valset.py` | ~15 held-out validation examples |

## Customizing

To adapt this for a different brand:
1. Edit `brand/guidelines.py` — swap in your brand's rules, banned phrases, and tone
2. Edit `brand/trainset.py` and `brand/valset.py` — write new labeled examples
3. Run `python demo.py`

The DSPy pipeline, metrics, and demo script don't need to change.
