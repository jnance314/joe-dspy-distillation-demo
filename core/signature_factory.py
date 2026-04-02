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
