import json
import logging
from typing import Tuple, Any, Dict

import click
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Metric
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, RobertaForSequenceClassification

from hs_generalization.train import get_dataloader
from hs_generalization.utils import load_config, get_dataset, plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


class ClassToLabel:
    def __init__(self, hate_label: int, benign_label: int):
        """Initialize the tracker of the best epoch."""
        self.hate_label = hate_label
        self.benign_label = benign_label

        self.class_to_label = {
            "hateful": self.hate_label,
            "non-hateful": self.benign_label,
        }

    def convert(self, example):
        example["labels"] = self.class_to_label[example["labels"]]
        return example


def evaluate_data(
        model: Any,
        dataloader: DataLoader,
        metric: Metric,
        device: str,
) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """Function for running model on evaluation or test set.
    In this function, the model loaded from checkpoint is applied on the evaluation or test set from a dataloader.
    Loss and accuracy are tracked as well.
    Args:
        model (Model): Model that is being trained.:
        dataloader (DataLoader): Object that will load the training data.
        metric (Metric): Metric that is being tracked.
        device (str): Device on which training will be done.
    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_accuracy (float): Average accuracy over the whole validation set.
    """
    model.eval()

    predictions = torch.tensor([])
    references = torch.tensor([])
    losses = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            references = torch.cat([references, batch["labels"].to("cpu")])

            losses.append(outputs.loss.detach().cpu().numpy())

    eval_loss = np.mean(losses)
    score = metric.compute(predictions=predictions, references=references)

    cm = confusion_matrix(references, predictions)

    results = {
        "loss": float(eval_loss),
        "confusion_matrix": cm
    }

    results = results | score
    return results, predictions, references


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

    accelerator = Accelerator()
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = "Paul/hatecheck"
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device", device)
    padding = config["processing"]["padding"]
    hate_speech_label = config["task"]["hate_speech_label"]
    benign_label = config["task"]["benign_label"]

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
    )
    class_to_label_converter = ClassToLabel(hate_speech_label, benign_label)
    dataset = dataset["test"].map(class_to_label_converter.convert)

    batch_size = config["pipeline"]["batch_size"]

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model_state_dict = {k.replace("module.", ""): v for (k, v) in checkpoint["model"].items()}
    model.load_state_dict(model_state_dict, strict=False)
    metric = evaluate.load("accuracy")

    logger.info(f" Device used: {device}.")
    logger.info(" Starting evaluating model on the data.")

    functionalities = list(set(dataset["functionality"]))
    functionality_to_results = {}
    for functionality in functionalities:
        logger.info(f"Evaluating functionality: {functionality}")

        subdataset = dataset.filter(lambda example: example["functionality"] == functionality)
        dataloader = get_dataloader(subdataset, tokenizer, batch_size, padding)

        results, predictions, references = evaluate_data(model, dataloader, metric, device)

        logger.info(f"Loss: {results['loss']}, Accuracy: {results['accuracy']}")
        plot_confusion_matrix(functionality, results['confusion_matrix'])

        correct_subdataset = subdataset.filter(
            lambda example,
            idx: references[idx] == predictions[idx],
            with_indices=True,
        )
        incorrect_subdataset = subdataset.filter(
            lambda example,
            idx: references[idx] != predictions[idx],
            with_indices=True,
        )
        results["confusion_matrix"] = results["confusion_matrix"].tolist()
        functionality_to_results[functionality] = {
            "results": results,
            "predictions": predictions.int().tolist(),
            "references": references.int().tolist(),
            "correct": correct_subdataset["test_case"],
            "incorrect": incorrect_subdataset["test_case"],
        }

    with open(config["pipeline"]["output_path"], "w") as f:
        json.dump(functionality_to_results, f, indent=2)


if __name__ == "__main__":
    main()
