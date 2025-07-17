import torch
import numpy as np
from loguru import logger
from sklearn.metrics import matthews_corrcoef

from finetuning.data_utils import load_and_prepare_data
from finetuning.model_utils import get_bert_sequence_classifier


def evaluate_on_test(model_path=None, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (test set)
    _, _, test_dataloader, tokenizer, num_labels, (X_test, y_test) = (
        load_and_prepare_data(batch_size=batch_size)
    )

    # Load model (placeholder: re-initialize, or load from model_path if provided)
    model = get_bert_sequence_classifier(num_labels=num_labels)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Starting test set evaluation...")
    predictions, true_labels = [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    # Matthew's correlation coefficient for each batch
    matthews_set = []
    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(
            true_labels[i], np.argmax(predictions[i], axis=1).flatten()
        )
        matthews_set.append(matthews)

    # Flatten for aggregate evaluation
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    test_mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    logger.info(
        "Test Matthew's correlation coefficient (MCC) using BERT Fine Tuning: {0:0.4f}".format(
            test_mcc
        )
    )


if __name__ == "__main__":
    evaluate_on_test()
