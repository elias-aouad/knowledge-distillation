from model_utils import load_models, create_optimizer
from train import train_one_epoch
from validate import validate
from utils import read_best_score, save_best_score, save_model

import torch
from datetime import datetime, timedelta
from time import time
from loguru import logger


def main():
    # === Configurations and Hyperparameters ===
    model_name = "student_model"  # TODO: Set your model name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    num_heads = 12  # TODO: Set according to your model
    alpha_L1, alpha_CE, alpha_KL, alpha_COS = 2, 1, 2, 1
    temperature = 2
    PATH = "student_model_best.pt"  # TODO: Set your save path

    # === Load models and optimizer ===
    student, teacher = load_models()
    # TODO: Move models to device
    optimizer = create_optimizer(student)

    # === Get model config values (placeholders) ===
    s_intermediate_size = getattr(student.config, "intermediate_size", 768)
    t_intermediate_size = getattr(teacher.config, "intermediate_size", 768)
    s_vocab_size = getattr(student.config, "vocab_size", 30522)
    t_vocab_size = getattr(teacher.config, "vocab_size", 30522)
    assert (
        s_vocab_size == t_vocab_size
    ), f"student and teacher model have different vocab sizes ({s_vocab_size}, {t_vocab_size})"
    assert (
        s_intermediate_size == t_intermediate_size
    ), f"student and teacher model have different intermediate sizes ({s_intermediate_size}, {t_intermediate_size})"

    # === Best score tracking ===
    s_mlm_acc_max, best_score_path = read_best_score(model_name)
    train_loss_set = []
    student_mlm_distrib = []
    teacher_mlm_distrib = []
    val_mlm_acc = []

    for epoch in range(epochs):
        top = time()
        current_time = (datetime.now() + timedelta(hours=2)).strftime("%H:%M:%S")
        logger.info(f"\n EPOCH {epoch}")
        logger.info(f"started at {current_time}")

        # === Training ===
        train_loss, s_mlm_acc, t_mlm_acc, sim_s_t = train_one_epoch(
            student,
            teacher,
            optimizer,
            device,
            epoch,
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
        )
        student_mlm_distrib.append(s_mlm_acc)
        teacher_mlm_distrib.append(t_mlm_acc)
        train_time = timedelta(seconds=int(round(time() - top)))
        logger.info(f"Train execution time {str(train_time)}")
        logger.info(f"Train loss: {train_loss}")
        logger.info(f"student MLM train accuracy : {round(100*s_mlm_acc, 1)} %")
        logger.info(f"teacher MLM train accuracy : {round(100*t_mlm_acc, 1)} %")
        logger.info(
            f"student-teacher train similarity in MLM : {round(100*sim_s_t, 1)} %"
        )

        # === Validation ===
        val_s_mlm_acc = validate(student, device, s_vocab_size)
        val_mlm_acc.append(val_s_mlm_acc)
        val_time = timedelta(seconds=int(round(time() - top))) - train_time
        logger.info(f"Val. execution time {str(val_time)}")
        logger.info(f"student MLM val. accuracy : {round(100*val_s_mlm_acc, 1)} %")

        if val_s_mlm_acc > s_mlm_acc_max:
            logger.info("hooray new record !!")
            save_model(student, PATH)
            s_mlm_acc_max = s_mlm_acc
            save_best_score(best_score_path, s_mlm_acc_max)


if __name__ == "__main__":
    main()
