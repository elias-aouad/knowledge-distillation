import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def load_and_prepare_data(
    tokenizer_name="bert-base-uncased",
    batch_size=16,
    max_length=512,
    test_size=0.1,
    random_state=42,
):
    # Load data
    twenty_train = fetch_20newsgroups(subset="train", shuffle=False)
    twenty_test = fetch_20newsgroups(subset="test", shuffle=True)

    # Split train into train/dev
    X_train, X_dev, y_train, y_dev = train_test_split(
        twenty_train["data"],
        twenty_train["target"],
        test_size=test_size,
        random_state=random_state,
    )

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_seq = tokenizer(
        X_train,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    dev_seq = tokenizer(
        X_dev,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    train_labels = torch.tensor(y_train)
    dev_labels = torch.tensor(y_dev)

    # DataLoaders
    train_data = TensorDataset(
        train_seq["input_ids"], train_seq["attention_mask"], train_labels
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    validation_data = TensorDataset(
        dev_seq["input_ids"], dev_seq["attention_mask"], dev_labels
    )
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=batch_size
    )

    # Prepare test set
    X_test, y_test = twenty_test["data"], twenty_test["target"]
    test_seq = tokenizer(
        X_test,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    test_labels = torch.tensor(y_test)
    test_data = TensorDataset(
        test_seq["input_ids"], test_seq["attention_mask"], test_labels
    )
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        tokenizer,
        len(twenty_train["target_names"]),
        (X_test, y_test),
    )
