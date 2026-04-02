"""
Model catalog with pricing data.

Pricing is hardcoded based on published rates as of April 2026.
For models with tiered pricing (e.g. Gemini Pro <=200k vs >200k),
we use the standard tier (<=200k context).
"""

MODEL_CATALOG = [
    # ── Google Gemini ──────────────────────────────────────────────
    {"id": "gemini/gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro", "provider": "google",
     "tier": "teacher", "input_cost": 2.00, "output_cost": 12.00},
    {"id": "gemini/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "google",
     "tier": "teacher", "input_cost": 1.25, "output_cost": 10.00},
    {"id": "gemini/gemini-3-flash-preview", "name": "Gemini 3 Flash", "provider": "google",
     "tier": "both", "input_cost": 0.50, "output_cost": 3.00},
    {"id": "gemini/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "google",
     "tier": "both", "input_cost": 0.30, "output_cost": 2.50},
    {"id": "gemini/gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite", "provider": "google",
     "tier": "student", "input_cost": 0.25, "output_cost": 1.50},
    {"id": "gemini/gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite", "provider": "google",
     "tier": "student", "input_cost": 0.10, "output_cost": 0.40},

    # ── OpenAI ─────────────────────────────────────────────────────
    # GPT-5.4 family (March 2026)
    {"id": "openai/gpt-5.4", "name": "GPT-5.4", "provider": "openai",
     "tier": "teacher", "input_cost": 2.50, "output_cost": 15.00},
    {"id": "openai/gpt-5.4-mini", "name": "GPT-5.4 Mini", "provider": "openai",
     "tier": "both", "input_cost": 0.75, "output_cost": 4.50},
    {"id": "openai/gpt-5.4-nano", "name": "GPT-5.4 Nano", "provider": "openai",
     "tier": "student", "input_cost": 0.20, "output_cost": 1.25},
    # GPT-4.1 family
    {"id": "openai/gpt-4.1", "name": "GPT-4.1", "provider": "openai",
     "tier": "teacher", "input_cost": 2.00, "output_cost": 8.00},
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "openai",
     "tier": "both", "input_cost": 0.40, "output_cost": 1.60},
    {"id": "openai/gpt-4.1-nano", "name": "GPT-4.1 Nano", "provider": "openai",
     "tier": "student", "input_cost": 0.05, "output_cost": 0.20},
    # GPT-4o family
    {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openai",
     "tier": "teacher", "input_cost": 2.50, "output_cost": 10.00},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai",
     "tier": "student", "input_cost": 0.15, "output_cost": 0.60},
    # Reasoning models
    {"id": "openai/o3", "name": "o3", "provider": "openai",
     "tier": "teacher", "input_cost": 2.00, "output_cost": 8.00},
    {"id": "openai/o4-mini", "name": "o4-mini", "provider": "openai",
     "tier": "both", "input_cost": 0.55, "output_cost": 2.20},

    # ── Anthropic ──────────────────────────────────────────────────
    # Claude 4.6 family (latest)
    {"id": "anthropic/claude-opus-4-6", "name": "Claude Opus 4.6", "provider": "anthropic",
     "tier": "teacher", "input_cost": 5.00, "output_cost": 25.00},
    {"id": "anthropic/claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "provider": "anthropic",
     "tier": "teacher", "input_cost": 3.00, "output_cost": 15.00},
    {"id": "anthropic/claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "anthropic",
     "tier": "both", "input_cost": 1.00, "output_cost": 5.00},
    # Claude 4.5 family
    {"id": "anthropic/claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "anthropic",
     "tier": "teacher", "input_cost": 3.00, "output_cost": 15.00},
    # Claude 3.5 Haiku (cheapest Anthropic)
    {"id": "anthropic/claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "provider": "anthropic",
     "tier": "student", "input_cost": 0.80, "output_cost": 4.00},
]


def get_model_cost(model_id: str) -> dict | None:
    """Look up cost for a model ID. Returns {"input_cost": float, "output_cost": float} or None."""
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            return {"input_cost": m["input_cost"], "output_cost": m["output_cost"]}
    return None


def get_teacher_models() -> list[dict]:
    return [m for m in MODEL_CATALOG if m["tier"] in ("teacher", "both")]


def get_student_models() -> list[dict]:
    return [m for m in MODEL_CATALOG if m["tier"] in ("student", "both")]
