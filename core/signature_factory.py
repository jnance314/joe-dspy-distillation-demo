"""
Dynamically builds dspy.Signature classes from TaskConfig field definitions.
"""

import dspy

from core.task_config import TaskConfig


def build_signature(task_config: TaskConfig) -> type:
    """Create a dspy.Signature subclass from the task's field definitions.

    Always includes a 'guidelines' InputField automatically.
    The Signature docstring is set to task_config.description.
    """
    namespace = {
        "__doc__": task_config.description,
        "__annotations__": {},
        "guidelines": dspy.InputField(desc="Task guidelines and rules"),
    }
    namespace["__annotations__"]["guidelines"] = str

    for f in task_config.fields:
        if f.field_type == "input":
            namespace[f.name] = dspy.InputField(desc=f.description)
        else:
            namespace[f.name] = dspy.OutputField(desc=f.description)
        namespace["__annotations__"][f.name] = str

    return type("DynamicSignature", (dspy.Signature,), namespace)


def build_module(task_config: TaskConfig) -> dspy.Module:
    """Create a dspy.Module that wraps ChainOfThought with the dynamic signature.

    The module bakes in the guidelines so the optimizer only needs to pass
    the input fields from the examples.
    """
    sig_class = build_signature(task_config)
    guidelines = task_config.guidelines

    class DynamicModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.check = dspy.ChainOfThought(sig_class)

        def forward(self, **kwargs):
            return self.check(guidelines=guidelines, **kwargs)

    return DynamicModule()
