import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings.input_ids)


def get_nli_dataset(config, tokenizer):

    if config.data_path:
        train_df = pd.read_csv(config.data_path + "/train.csv")
        train_premises = train_df["sentence1"]
        train_hypotheses = train_df["sentence2"]
        train_labels = train_df["gold_label"]

        dev_df = pd.read_csv(config.data_path + "/dev.csv")
        dev_premises = dev_df["sentence1"]
        dev_hypotheses = dev_df["sentence2"]
        dev_labels = dev_df["gold_label"]

        test_df = pd.read_csv(config.data_path + "/test.csv")
        test_premises = test_df["sentence1"]
        test_hypotheses = test_df["sentence2"]
        test_labels = test_df["gold_label"]
    else:
        train_dataset = datasets.load_dataset("xnli", config.train_language, split="train")
        train_premises = train_dataset["premise"]
        train_hypotheses = train_dataset["hypothesis"]
        train_labels = train_dataset["label"]

        dev_dataset = datasets.load_dataset(
            "xnli", config.test_language, split="validation"
        )
        dev_premises = dev_dataset["premise"]
        dev_hypotheses = dev_dataset["hypothesis"]
        dev_labels = dev_dataset["label"]

        test_dataset = datasets.load_dataset("xnli", config.test_language, split="test")
        test_premises = test_dataset["premise"]
        test_hypotheses = test_dataset["hypothesis"]
        test_labels = test_dataset["label"]

    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        padding=True,
    )
    dev_encodings = tokenizer(
        dev_premises, dev_hypotheses, truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_premises,
        test_hypotheses,
        truncation=True,
        padding=True,
    )

    train_dataset = NLIDataset(train_encodings, train_labels)
    dev_dataset = NLIDataset(dev_encodings, dev_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader
