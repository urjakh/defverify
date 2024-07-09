import logging
from typing import Tuple, Any, Dict

import click
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification

from hs_generalization.train import get_dataloader
from hs_generalization.utils import load_config, get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


def evaluate_data(
        model: Any,
        dataloader: DataLoader,
        device: str,
        dataset_name: str,
        model_hate_label: int,
        data_hate_label: int,
) -> Tuple[np.float_, Dict, list, Any]:
    """Function for running model on evaluation or test set.
    In this function, the model loaded from checkpoint is applied on the evaluation or test set from a dataloader.
    Loss and accuracy are tracked as well.
    Args:
        model (Model): Model that is being trained.:
        dataloader (DataLoader): Object that will load the training data.
        device (str): Device on which training will be done.
    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_accuracy (float): Average accuracy over the whole validation set.
    """
    model.eval()

    predictions = torch.tensor([])
    references = torch.tensor([])

    with torch.no_grad():
        losses = []
        all_predictions = []
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if dataset_name in ["founta", "mathew", "davidson"]:
                labels = torch.where(labels > 1, 1, labels)
            if dataset_name == "talat_hovy":
                # 2 is the benign label, so we change all the non-benign labels to hate label of data for consistency
                labels = torch.where(labels != 2, data_hate_label, labels)
                labels = torch.where(labels == 2, 1, labels)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            references = torch.cat([references, batch["labels"].to("cpu")])

            losses.append(outputs.loss.detach().cpu().numpy())
            all_predictions.extend(predictions.tolist())

    recognized_hate = (predictions == model_hate_label) & (references == data_hate_label)

    hate_detected = torch.sum(recognized_hate)
    total_hate_in_data = torch.sum(references == data_hate_label)
    print(hate_detected, total_hate_in_data)
    accuracy = hate_detected / total_hate_in_data
    return accuracy


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
def main(config_path: str):
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        config_path (str): path to the config file for the training experiment.
    """
    config = load_config(config_path)
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"])

    accelerator = Accelerator()
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    dataset_directory = config["task"].get("dataset_directory")
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device", device)
    padding = config["processing"]["padding"]
    model_hate_label = config["task"]["model_hate_label"]
    data_hate_label = config["task"]["data_hate_label"]

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
        dataset_directory=dataset_directory,
    )
    dataset = dataset["test"]
    batch_size = config["pipeline"]["batch_size"]
    dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)

    # Load metric, model, optimizer, and learning rate scheduler.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    logger.info(f" Device: {device}.")
    logger.info(" Starting evaluating model on the data.")
    accuracy = evaluate_data(model, dataloader, device, dataset_name, model_hate_label, data_hate_label)
    logger.info(f"Accuracy: {accuracy}")

    wandb.log({"accuracy": accuracy})


if __name__ == "__main__":
    main()
