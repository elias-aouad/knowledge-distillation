import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


def get_extended_attention_mask(attention_mask, input_shape, device=torch.device('cpu')):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    # dtype = float
    extended_attention_mask = extended_attention_mask.float()  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def train_one_epoch(
    tokenizer,
    student,
    teacher,
    optimizer,
    device,
    num_heads,
    s_intermediate_size,
    t_intermediate_size,
    s_vocab_size,
    t_vocab_size,
    alpha_L1,
    alpha_CE,
    alpha_KL,
    alpha_COS,
    temperature,
    train_loss_set,
):
    student.train()
    teacher.eval()
    criterion_L1 = nn.L1Loss()
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction="batchmean")
    criterion_COS = nn.CosineEmbeddingLoss(reduction="mean")

    s_true_preds, t_true_preds, sim_with_teacher, nb_samples, nb_tr_steps, tr_loss = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    train_dataloaders = generate_new_dataloaders(tokenizer, dataset, batch_size=8)

    for train_dataloader in train_dataloaders:
        for step, batch in enumerate(train_dataloader):
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
            optimizer.zero_grad()
            loss = 0
            t_hidden_state = teacher.bert.embeddings(b_input_ids).detach().data
            s_hidden_state = student.bert.embeddings(b_input_ids)
            t_attention_mask = (
                get_extended_attention_mask(b_input_mask, t_hidden_state.shape)
                .detach()
                .data
            )
            s_attention_mask = get_extended_attention_mask(
                b_input_mask, s_hidden_state.shape
            )
            for i in range(12):
                use_teacher_forcing = np.random.choice([True, False])
                t_attention_output, t_attention_matrices = teacher.bert.encoder.layer[
                    i
                ].attention(
                    t_hidden_state,
                    attention_mask=t_attention_mask,
                    output_attentions=True,
                )
                t_attention_output, t_attention_matrices = (
                    t_attention_output.detach().data,
                    t_attention_matrices.detach().data,
                )
                s_attention_output, s_attention_matrices = student.bert.encoder.layer[
                    i
                ].attention(
                    s_hidden_state,
                    attention_mask=s_attention_mask,
                    output_attentions=True,
                )
                loss += alpha_L1 * criterion_L1(
                    s_attention_matrices, t_attention_matrices[:, :num_heads]
                )
                t_intermediate_output = (
                    teacher.bert.encoder.layer[i]
                    .intermediate.dense(t_attention_output)
                    .detach()
                    .data
                )
                s_intermediate_output = student.bert.encoder.layer[
                    i
                ].intermediate.dense(s_attention_output)
                target_size = s_intermediate_output.view(-1, s_intermediate_size).size(
                    0
                )
                target = (
                    s_intermediate_output.view(-1, s_intermediate_size)
                    .new(target_size)
                    .fill_(1)
                )
                loss += alpha_COS * criterion_COS(
                    s_intermediate_output.view(-1, s_intermediate_size),
                    t_intermediate_output.view(-1, t_intermediate_size),
                    target,
                )
                if use_teacher_forcing:
                    s_intermediate_output = deepcopy(t_intermediate_output)
                    s_intermediate_output.requires_grad = False
                t_intermediate_output = (
                    teacher.bert.encoder.layer[i]
                    .intermediate.intermediate_act_fn(t_intermediate_output)
                    .detach()
                    .data
                )
                s_intermediate_output = student.bert.encoder.layer[
                    i
                ].intermediate.intermediate_act_fn(s_intermediate_output)
                t_hidden_state = (
                    teacher.bert.encoder.layer[i]
                    .output(t_intermediate_output, t_attention_output)
                    .detach()
                    .data
                )
                s_hidden_state = student.bert.encoder.layer[i].output(
                    s_intermediate_output, s_attention_output
                )
            t_logits = teacher.cls(t_hidden_state).view(-1, t_vocab_size).detach().data
            s_logits = student.cls(s_hidden_state).view(-1, s_vocab_size)
            loss += (
                alpha_KL
                * criterion_KL(
                    F.log_softmax(s_logits / temperature, dim=-1),
                    F.softmax(t_logits / temperature, dim=-1),
                )
                * (temperature) ** 2
            )
            s_logits_mlm = s_logits.view(b_size, -1, s_vocab_size)
            s_logits_mlm = s_logits_mlm[range(b_size), position_token]
            loss += alpha_CE * criterion_CE(s_logits_mlm, target_token)
            t_logits_mlm = t_logits.view(b_size, -1, t_vocab_size)
            t_logits_mlm = t_logits_mlm[range(b_size), position_token]
            loss.backward()
            optimizer.step()
            s_predicted_mlm = s_logits_mlm.argmax(axis=-1)
            s_true_preds += s_predicted_mlm.eq(target_token).sum().item()
            t_predicted_mlm = t_logits_mlm.argmax(axis=-1)
            t_true_preds += t_predicted_mlm.eq(target_token).sum().item()
            sim_with_teacher += s_predicted_mlm.eq(t_predicted_mlm).sum().item()
            nb_samples += b_size
            train_loss_set.append(loss.item())
            tr_loss += loss.item()
            nb_tr_steps += 1
    train_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
    s_mlm_acc = s_true_preds / nb_samples if nb_samples > 0 else 0
    t_mlm_acc = t_true_preds / nb_samples if nb_samples > 0 else 0
    sim_s_t = sim_with_teacher / nb_samples if nb_samples > 0 else 0
    return train_loss, s_mlm_acc, t_mlm_acc, sim_s_t
