"""
Model catalog with pricing data.

Pricing is hardcoded based on published rates as of April 2026.
For Gemini Pro models, we use the <=200k tier (most common for this use case).
"""

MODEL_CATALOG = [
    # Gemini
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

    # OpenAI
    {"id": "openai/gpt-4.1", "name": "GPT-4.1", "provider": "openai",
     "tier": "teacher", "input_cost": 2.00, "output_cost": 8.00},
    {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openai",
     "tier": "teacher", "input_cost": 2.50, "output_cost": 10.00},
    {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "openai",
     "tier": "both", "input_cost": 0.40, "output_cost": 1.60},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai",
     "tier": "student", "input_cost": 0.15, "output_cost": 0.60},
    {"id": "openai/gpt-4.1-nano", "name": "GPT-4.1 Nano", "provider": "openai",
     "tier": "student", "input_cost": 0.05, "output_cost": 0.20},

    # Anthropic
    {"id": "anthropic/claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "anthropic",
     "tier": "teacher", "input_cost": 3.00, "output_cost": 15.00},
    {"id": "anthropic/claude-haiku-3-5-20241022", "name": "Claude Haiku 3.5", "provider": "anthropic",
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
