import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import trange
import numpy as np
from time import time
from loguru import logger

from finetuning.data_utils import load_and_prepare_data
from finetuning.model_utils import get_bert_sequence_classifier


def train(
    epochs=4,
    batch_size=16,
    lr=2e-5,
    use_scheduler=False,
    device=None,
    save_model_path=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data and model
    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        tokenizer,
        num_labels,
        _,
    ) = load_and_prepare_data(batch_size=batch_size)
    model = get_bert_sequence_classifier(num_labels=num_labels)
    model.to(device)

    # Optimizer and loss
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not "bias" in n],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in param_optimizer if "bias" in n], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    criterion = nn.CrossEntropyLoss()

    if use_scheduler:
        num_warmup_steps, num_training_steps = 100, 1000
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    train_loss_set = []
    top = time()

    for epoch in trange(epochs, desc="Epoch"):
        logger.info(f"EPOCH {epoch}")
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        true_preds, nb_samples = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(logits, b_labels)
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        logger.info(f"Train loss: {tr_loss/nb_tr_steps}")

        # VALIDATION
        model.eval()
        true_preds, nb_samples = 0, 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, attention_mask=b_input_mask)
            predicted_labels = logits.argmax(axis=-1)
            true_preds += predicted_labels.eq(b_labels).sum().item()
            nb_samples += b_labels.shape[0]
        logger.info(f"Validation MCC : {true_preds/nb_samples}")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    training_time = time() - top
    logger.info(f"Total trainable parameters: {num_parameters}")
    logger.info(f"Training time: {training_time:.2f} seconds")

    # Optionally save the model
    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
        logger.info(f"Model saved to {save_model_path}")


if __name__ == "__main__":
    train()
