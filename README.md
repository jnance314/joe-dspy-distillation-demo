# DSPy Prompt Optimizer

Automatic prompt optimization with DSPy's MIPROv2. Comes with a web UI and a CLI demo.

Optimize once with a smart model, deploy on a cheap model. No manual prompt engineering.

## Quick start

```bash
uv sync
cp .env.example .env
# Add your API keys to .env (Gemini, OpenAI, and/or Anthropic)
```

### Web UI

```bash
uv run run_web.py
# Open http://localhost:8000
```

### CLI demo

```bash
uv run demo.py
```

## Deploy to Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/dspy-optimizer

# Deploy with API keys as secrets
gcloud run deploy dspy-optimizer \
  --image gcr.io/YOUR_PROJECT/dspy-optimizer \
  --set-env-vars GEMINI_API_KEY=xxx,OPENAI_API_KEY=xxx,ANTHROPIC_API_KEY=xxx \
  --memory 2Gi \
  --timeout 900 \
  --allow-unauthenticated
```

Or use the `--set-secrets` flag to pull from Secret Manager instead of inline env vars.

## What it does

1. A cheap model tries your task zero-shot (naive baseline)
2. An expensive "teacher" model generates high-quality demonstrations
3. MIPROv2 finds the optimal prompt (instructions + few-shot examples) for the cheap model
4. The cheap model with the optimized prompt matches or beats the expensive model

## Default demo: Brand voice compliance (Liquid Death)

Given marketing copy and brand guidelines, the system:
- Labels copy as compliant or non-compliant
- Flags specific problematic phrases
- Suggests on-brand replacements

## Files

| Path | Purpose |
|------|---------|
| `run_web.py` | Web UI entry point (FastAPI + Alpine.js) |
| `demo.py` | CLI demo (standalone, no web server) |
| `core/` | Task-agnostic DSPy engine |
| `server/` | FastAPI backend |
| `static/` | Frontend (no build step) |
| `brand/` | Default task data (Liquid Death) |
| `tasks/` | Saved task configs (JSON) |
| `Dockerfile` | Cloud Run / container deployment |
| `GUIDE.md` | User guide with walkthroughs |

## Supported model providers

Gemini, OpenAI, and Anthropic models are pre-configured in the dropdown.
Add your API keys to `.env` for the providers you want to use.
DSPy uses LiteLLM under the hood, so any LiteLLM-compatible model works.
