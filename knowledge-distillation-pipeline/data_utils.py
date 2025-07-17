from torch.utils.data.sampler import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from random import sample
from datasets import load_dataset


dataset = load_dataset("bookcorpus")
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


def get_split_dataset(split="train"):
    if split == "train":
        return train_dataset
    elif split == "val":
        return val_dataset
    else:
        raise ValueError("split must be 'train' or 'val'")


def generate_new_dataloaders(
    tokenizer, max_n_samples=25_000, batch_size=16, split="train"
):
    data_split = get_split_dataset(split)
    len_dataset = len(data_split)
    indices = sample(range(len_dataset), k=min(max_n_samples, len_dataset))

    text_corpus = data_split[indices]["text"]
    selected_data = text_corpus[:max_n_samples]
    selected_data = sorted(selected_data, key=lambda sentence: len(sentence.split()))

    train_dataloaders = []
    for i in range(0, len(selected_data), 5_000):
        chunk_of_data = selected_data[i : i + 5_000]
        kd_seq = tokenizer(
            chunk_of_data, padding="longest", truncation=True, return_tensors="pt"
        )
        train_data = TensorDataset(kd_seq["input_ids"], kd_seq["attention_mask"])
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        train_dataloaders += [train_dataloader]
    shuffle(train_dataloaders)
    return train_dataloaders
