from argparse import ArgumentParser
from data import get_nli_dataset
import logging
import models
import numpy as np
from pathlib import Path
import random
import torch
from tqdm import tqdm
from transformers import AdamW


parser = ArgumentParser(description="XNLI with Transformers")

parser.add_argument("--train_language", type=str, default=None)
parser.add_argument("--test_language", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--output_path", type=str, default="output")
parser.add_argument(
    "--model",
    type=str,
    choices=["bert", "xlmroberta"],
    default="xlmroberta",
)
parser.add_argument("--data_path", type=str)

logging.basicConfig(level=logging.INFO)


def train(config, train_loader, model, optim, device, epoch):
    logging.info("Starting training...")
    model.train()
    logging.info(f"Epoch: {epoch + 1}/{config.epochs}")
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        if i == 0 or i % config.log_every == 0 or i + 1 == len(train_loader):
            logging.info(
                "Epoch: {} - Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:<.4f}".format(
                    epoch + 1,
                    100.0 * (1 + i) / len(train_loader),
                    i + 1,
                    len(train_loader),
                    loss.item(),
                )
            )


def evaluate(model, dataloader, device):
    logging.info("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = preds[1].argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch["labels"].cpu().numpy())

    logging.info("Done evaluation")
    return np.concatenate(eval_labels), np.concatenate(eval_preds)


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = (
        torch.device(f"cuda:{config.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    logging.info(f"Training on {device}.")

    tokenizer, model = models.get_model(config)

    train_loader, dev_loader, test_loader = get_nli_dataset(config, tokenizer)

    optim = AdamW(model.parameters(), lr=config.learning_rate)
    model.to(device)

    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        train(config, train_loader, model, optim, device, epoch)
        dev_labels, dev_preds = evaluate(model, dev_loader, device)

        dev_accuracy = (dev_labels == dev_preds).mean()
        logging.info(f"Dev accuracy after epoch {epoch+1}: {dev_accuracy}")

        snapshot_path = f"{config.output_path}/{config.model}-mnli_snapshot_epoch_{epoch+1}_devacc_{round(dev_accuracy, 3)}.pt"
        torch.save(model, snapshot_path)

    test_labels, test_preds = evaluate(model, test_loader, device)
    test_accuracy = (test_labels == test_preds).mean()

    logging.info(f"Test accuracy for model {config.model}: {test_accuracy}")

    with open(
        f"{config.output_path}/{config.model}.results.txt",
        "w",
    ) as resultfile:
        resultfile.write(f"Test accuracy: {test_accuracy}")

    final_snapshot_path = f"{config.output_path}/{config.model}-mnli_final_snapshot_epochs_{config.epochs}_devacc_{round(dev_accuracy, 3)}.pt"
    torch.save(model, final_snapshot_path)


if __name__ == "__main__":
    main()
