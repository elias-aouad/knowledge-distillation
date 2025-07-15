import torch
from torch.optim import AdamW


# Placeholder: Replace with actual model loading logic
def load_models():
    """Load and return student and teacher models."""
    student = None  # TODO: Load student model
    teacher = None  # TODO: Load teacher model
    return student, teacher


def create_optimizer(student, weight_decay=0.01, lr=2e-5):
    """Create optimizer for the student model."""
    param_optimizer = list(student.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not "bias" in n],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in param_optimizer if "bias" in n], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    return optimizer
