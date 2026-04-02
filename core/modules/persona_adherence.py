"""Single-stage persona adherence module."""

import dspy
from core.signature_factory import build_signature


class PersonaAdherenceModule(dspy.Module):
    def __init__(self, task_config):
        super().__init__()
        sig_class = build_signature(task_config)
        self.guidelines = task_config.guidelines
        self.check = dspy.ChainOfThought(sig_class)

    def forward(self, **kwargs):
        return self.check(guidelines=self.guidelines, **kwargs)
