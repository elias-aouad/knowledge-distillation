import torch
import numpy as np


def generate_new_dataloaders(max_n_samples=5000, batch_size=8):
    # TODO: Implement data loader generation
    return []


def validate(student, device, s_vocab_size):
    student.eval()
    val_dataloaders = generate_new_dataloaders(max_n_samples=5000, batch_size=8)
    nb_val_samples, val_s_true_preds = 0, 0
    for val_dataloader in val_dataloaders:
        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            b_size = b_input_ids.shape[0]
            target_token = torch.zeros(b_size).int().long().to(device)
            position_token = [0] * b_size
            for i in range(b_size):
                idx_i = (
                    torch.where(
                        (b_input_ids[i] != 0)
                        & (b_input_ids[i] != 101)
                        & (b_input_ids[i] != 102)
                    )[0]
                    .cpu()
                    .tolist()
                )
                j = np.random.choice(idx_i)
                target_token[i] = b_input_ids[i, j]
                b_input_ids[i, j] = 103
                position_token[i] = j
            s_logits = student(b_input_ids, attention_mask=b_input_mask)["logits"]
            s_logits_mlm = s_logits.view(b_size, -1, s_vocab_size)
            s_logits_mlm = s_logits_mlm[range(b_size), position_token]
            s_predicted_mlm = s_logits_mlm.argmax(axis=-1)
            val_s_true_preds += s_predicted_mlm.eq(target_token).sum().item()
            nb_val_samples += b_size
    val_s_mlm_acc = val_s_true_preds / nb_val_samples if nb_val_samples > 0 else 0
    return val_s_mlm_acc
