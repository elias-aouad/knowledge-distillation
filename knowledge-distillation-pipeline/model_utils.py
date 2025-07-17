import torch
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM


def load_student_model(hidden_size: int, num_layers: int, num_heads: int):
    """Load the student model."""
    config = BertConfig(
        hidden_size=hidden_size, 
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads
    )
    student = BertForMaskedLM(config)
    return student


def load_teacher_model():
    """Load the teacher model."""
    teacher = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return teacher


def load_models(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
):
    """Load and return student and teacher models."""
    student = load_student_model(hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads)
    teacher = load_teacher_model()
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
