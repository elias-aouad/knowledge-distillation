# Knowledge Distillation for BERT Pretraining and Benchmarking

This project provides a pipeline for pretraining BERT models using knowledge distillation, followed by fine-tuning and benchmarking on a classification task. The goal is to enable efficient training of compact BERT models (students) by leveraging the knowledge of larger, pretrained teacher models.

## Features

- **Knowledge Distillation Pipeline:** Pretrains a student BERT model using a teacher (BERT-base) on the BookCorpus dataset.
- **Fine-tuning:** Trains the distilled model on a downstream classification task (20 Newsgroups).
- **Benchmarking:** Evaluates and compares the performance of student and teacher models.

## Project Structure

```
knowledge-distillation/
  ├── knowledge-distillation-pipeline/   # Pretraining and distillation code
  └── finetuning/                       # Fine-tuning and evaluation code
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd knowledge-distillation
   ```

2. **Install dependencies:**
   ```bash
   uv init
   uv sync
   ```

   **Main dependencies:**
   - torch
   - transformers
   - datasets
   - scikit-learn
   - loguru
   - tqdm

   *(See `pyproject.toml` for full list.)*

## Usage

### 1. Pretrain Student Model with Knowledge Distillation

Run the distillation pipeline (pretraining student BERT from teacher):

```bash
cd knowledge-distillation-pipeline
python run.py
```

- Uses the BookCorpus dataset (downloaded automatically via `datasets`).
- Model checkpoints and best scores are saved in the working directory.

### 2. Fine-tune on Classification Task

After pretraining, fine-tune the student model on a classification task (20 Newsgroups):

```bash
cd ../finetuning
python train.py
```

- The script loads and tokenizes the 20 Newsgroups dataset.
- Trains a sequence classifier on the dataset.

### 3. Evaluate on Test Set

To evaluate the fine-tuned model:

```bash
python test.py
```

- Computes and logs the Matthews correlation coefficient (MCC) on the test set.

## Datasets

- **Pretraining:** [BookCorpus](https://huggingface.co/datasets/bookcorpus) (downloaded automatically)
- **Fine-tuning:** [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) (downloaded automatically)

## Customization

- You can adjust model size, number of layers, and other hyperparameters in `knowledge-distillation-pipeline/run.py`.
- To use a different dataset for fine-tuning, modify `finetuning/data_utils.py`.

## License

MIT License
