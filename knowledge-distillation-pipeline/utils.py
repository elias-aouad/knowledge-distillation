import os
import torch


def read_best_score(model_name):
    best_score_path = model_name + "-BEST-SCORE.txt"
    if best_score_path in os.listdir():
        with open(best_score_path, "r") as f:
            file_line = f.readlines()
            s_mlm_acc_max = float(file_line[0][:-1])
    else:
        s_mlm_acc_max = 0
    return s_mlm_acc_max, best_score_path


def save_best_score(best_score_path, s_mlm_acc_max):
    with open(best_score_path, "w") as f:
        f.write(str(s_mlm_acc_max) + "\n")


def save_model(student, path):
    torch.save(student.state_dict(), path)
