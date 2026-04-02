"""Task module registry.

Each task ships its own dspy.Module subclass. The engine calls
get_module(task_config) to instantiate the right one based on module_key.
"""

from core.modules.brand_voice import BrandVoiceModule
from core.modules.persona_adherence import PersonaAdherenceModule
from core.modules.research_synthesizer import ResearchSynthesizerModule

TASK_MODULES = {
    "brand_voice": BrandVoiceModule,
    "persona_adherence": PersonaAdherenceModule,
    "research_synthesizer": ResearchSynthesizerModule,
}


def get_module(task_config):
    """Instantiate the correct module for a task config."""
    key = getattr(task_config, "module_key", "brand_voice")
    if key not in TASK_MODULES:
        raise ValueError(f"Unknown module_key: {key!r}. Available: {list(TASK_MODULES.keys())}")
    return TASK_MODULES[key](task_config)
