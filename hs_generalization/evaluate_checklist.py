import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Any, Dict, Union, List

import click
import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Metric, Dataset
from evaluate import EvaluationModule
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, RobertaForSequenceClassification, AutoModelForSequenceClassification

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
        metric: Union[Metric, EvaluationModule],
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


class HSTypeFilters:
    def __init__(self, added_information: Dataset, to_evaluate: Union[str, List[str]] = "all"):
        if to_evaluate == "all":
            self.to_evaluate = []

        self.to_evaluate = to_evaluate
        if self.to_evaluate == "all":
            self.to_evaluate = [
                "non_and_hate", "functionalities", "target_types", "dominance", "explicit_references", "consequences"
            ]

        self.added_information = added_information
        self.target_types = list(set(added_information["target_type"]))
        self.dominance = list(set(added_information["dominance"]))
        self.explicit_references = list(set(added_information["explicit_ref"]))
        self.consequences = list(set(added_information["incites"]))
        self.functionalities = list(set(added_information["functionality"]))
        self.hate_types = ["non-hateful", "hateful"]

        self.accuracy = evaluate.load("accuracy")

        self.results = defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def split_on_type(type_split: str, value: str, added_information: Dataset, dataset: Dataset):
        type_split_to_key = {
            "hate_type": "label_gold",
            "functionality": "functionality",
            "target_type": "target_type",
            "dominance": "dominance",
            "perpetrator_characteristics": "in_group",
            "consequences": "incites",
            "target_identity": "target_ident",
            "explicit_reference": "explicit_ref"
        }
        key = type_split_to_key[type_split]

        added_information_updated = added_information.filter(lambda example: example[key] == value)
        case_ids = added_information_updated["case_id"]
        subdataset = dataset.filter(lambda example: example["case_id"] in case_ids)
        return case_ids, added_information_updated, subdataset

    def evaluate_non_and_hate(self, dataset: Dataset):
        logger.info("Evaluating hateful and non-hateful separately.")

        for hate_type in self.hate_types:
            case_ids, _, subdataset = self.split_on_type("hate_type", hate_type, self.added_information, dataset)
            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])

            logger.info(f"Accuracy {hate_type}: {accuracy}")
            self.results["hate_type"][hate_type] = {"case_ids": case_ids, "accuracy": accuracy["accuracy"]}

    def evaluate_functionalities(self, dataset: Dataset):
        logger.info("Evaluating all the HateCheck functionalities separately.")

        for functionality in self.functionalities:
            subdataset = dataset.filter(lambda example: example["functionality"] == functionality)
            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])

            logger.info(f"Accuracy {functionality}: {accuracy}")
            self.results["functionalities"][functionality] = {
                "case_ids": subdataset["case_id"],
                "accuracy": accuracy["accuracy"]
            }

    def evaluate_target_types(self, dataset: Dataset):
        logger.info("Evaluating for individual target types.")

        for target_type in self.target_types:
            case_ids, info, subdataset = self.split_on_type("target_type", target_type, self.added_information, dataset)
            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])

            logger.info(f"Accuracy {target_type}: {accuracy}")

            self.results["target_types"][target_type]["overall"] = {
                "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

            for target in list(set(subdataset["target_ident"])):
                case_ids, _, target_subdataset = self.split_on_type("target_identity", target, info, dataset)
                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy {target_type} - {target}: {accuracy}")
                self.results["target_types"][target_type][target] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            for hate_type in self.hate_types:
                case_ids, _, target_subdataset = self.split_on_type("hate_type", hate_type, info, dataset)
                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy {target_type} - {hate_type}: {accuracy}")
                self.results["target_types"][target_type][hate_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

    def evaluate_dominance(self, dataset: Dataset):
        logger.info("Evaluating dominance (per target type).")

        for dominance in self.dominance:
            case_ids, info, subdataset = self.split_on_type(
                "dominance", dominance, self.added_information, dataset
            )
            accuracy = self.accuracy.compute(
                predictions=subdataset["predictions"],
                references=subdataset["references"]
            )
            logger.info(f"Aggregate Accuracy Dominance: {dominance}: {accuracy}")
            self.results["dominance"][dominance]["overall"] = {
                "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

            self.results["dominance"][dominance]["target_type"] = {}

            for target_type in self.target_types:
                case_ids, _, target_subdataset = self.split_on_type("target_type", target_type, info, dataset)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Dominance: {dominance} for {target_type} : {accuracy}")
                self.results["dominance"][dominance]["target_type"][target_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            self.results["dominance"][dominance]["functionalities"] = {}

            for functionality in self.functionalities:
                target_subdataset = dataset.filter(lambda example: example["functionality"] == functionality)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Dominance: {dominance} for {functionality} : {accuracy}")
                self.results["dominance"][dominance]["functionalities"][functionality] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            for hate_type in self.hate_types:
                case_ids, _, target_subdataset = self.split_on_type("hate_type", hate_type, info, dataset)
                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy {dominance} - {hate_type}: {accuracy}")
                self.results["dominance"][dominance][hate_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

    def evaluate_perpetrator_characteristics(self, dataset: Dataset):
        logger.info("Evaluating Perpetrator Characteristics - Member of Own Group.")

        case_ids, info, subdataset = self.split_on_type(
            "perpetrator_characteristics", "yes", self.added_information, dataset
        )

        accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])
        logger.info(f"Accuracy Perpetrator Characteristics - Member of Own Group: {accuracy}")

        self.results["perpetrator_in_group"]["overall"] = {
            "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
        }

        for target_type in self.target_types:
            case_ids, _, target_subdataset = self.split_on_type(
                "target_type", target_type, self.added_information, dataset
            )

            accuracy = self.accuracy.compute(
                predictions=target_subdataset["predictions"],
                references=target_subdataset["references"]
            )
            logger.info(f"Accuracy Perpetrator Characteristics - Member of Own Group for {target_type} : {accuracy}")
            self.results["perpetrator_in_group"]["target_type"][target_type] = {
                "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

        for functionality in self.functionalities:
            target_subdataset = dataset.filter(lambda example: example["functionality"] == functionality)

            accuracy = self.accuracy.compute(
                predictions=target_subdataset["predictions"],
                references=target_subdataset["references"]
            )
            logger.info(f"Accuracy Perpetrator Characteristics - Member of Own Group for {functionality} : {accuracy}")
            self.results["perpetrator_in_group"]["functionalities"][functionality] = {
                "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

        for hate_type in self.hate_types:
            case_ids, _, target_subdataset = self.split_on_type("hate_type", hate_type, self.added_information, dataset)
            accuracy = self.accuracy.compute(
                predictions=target_subdataset["predictions"],
                references=target_subdataset["references"]
            )
            logger.info(f"Accuracy Perpetrator Characteristics - Member of Own Group - {hate_type}: {accuracy}")
            self.results["perpetrator_in_group"][hate_type] = {
                "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

    def evaluate_explicit_references(self, dataset: Dataset):
        logger.info("Evaluating Explicit References.")

        for explicit_ref in self.explicit_references:
            case_ids, info, subdataset = self.split_on_type(
                "explicit_reference", explicit_ref, self.added_information, dataset
            )

            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])
            logger.info(f"Accuracy Explicit Reference {explicit_ref}: {accuracy}")

            self.results["explicit_reference"][explicit_ref]["overall"] = {
                "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

            self.results["explicit_reference"][explicit_ref]["target_type"] = {}

            for target_type in self.target_types:
                case_ids, _, target_subdataset = self.split_on_type("target_type", target_type, info, dataset)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Explicit Reference {explicit_ref} for {target_type} : {accuracy}")
                self.results["explicit_reference"][explicit_ref]["target_type"][target_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            self.results["explicit_reference"][explicit_ref]["functionalities"] = {}

            for functionality in self.functionalities:
                target_subdataset = dataset.filter(lambda example: example["functionality"] == functionality)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Explicit Reference {explicit_ref} for {functionality} : {accuracy}")
                self.results["explicit_reference"][explicit_ref]["functionalities"][functionality] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            for hate_type in self.hate_types:
                case_ids, _, target_subdataset = self.split_on_type("hate_type", hate_type, info, dataset)
                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Explicit Reference {explicit_ref} - {hate_type}: {accuracy}")
                self.results["explicit_reference"][explicit_ref][hate_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

    def evaluate_consequences(self, dataset: Dataset):
        logger.info("Evaluating Consequences.")

        for consequence in self.consequences:
            case_ids, info, subdataset = self.split_on_type(
                "consequences", consequence, self.added_information, dataset
            )

            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])
            logger.info(f"Accuracy Consequence {consequence}: {accuracy}")

            self.results["consequences"][consequence]["overall"] = {
                "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

            self.results["consequences"][consequence]["target_type"] = {}

            for target_type in self.target_types:
                case_ids, _, target_subdataset = self.split_on_type("target_type", target_type, info, dataset)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Consequence {consequence} for {target_type} : {accuracy}")
                self.results["consequences"][consequence]["target_type"][target_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            self.results["consequences"][consequence]["functionalities"] = {}

            for functionality in self.functionalities:
                target_subdataset = dataset.filter(lambda example: example["functionality"] == functionality)

                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Consequence {consequence} for {functionality} : {accuracy}")
                self.results["consequences"][consequence]["functionalities"][functionality] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

            for hate_type in self.hate_types:
                case_ids, _, target_subdataset = self.split_on_type("hate_type", hate_type, info, dataset)
                accuracy = self.accuracy.compute(
                    predictions=target_subdataset["predictions"],
                    references=target_subdataset["references"]
                )
                logger.info(f"Accuracy Consequence {consequence} - {hate_type}: {accuracy}")
                self.results["consequences"][consequence][hate_type] = {
                    "case_ids": target_subdataset["case_id"], "accuracy": accuracy["accuracy"]
                }

    def evaluate(self, dataset: Dataset):
        for evaluate_type in self.to_evaluate:
            if evaluate_type == "target_types":
                self.evaluate_target_types(dataset)
            elif evaluate_type == "dominance":
                self.evaluate_dominance(dataset)
            elif evaluate_type == "explicit_references":
                self.evaluate_explicit_references(dataset)
            elif evaluate_type == "consequences":
                self.evaluate_consequences(dataset)
            elif evaluate_type == "non_and_hate":
                self.evaluate_non_and_hate(dataset)
            elif evaluate_type == "functionalities":
                self.evaluate_functionalities(dataset)


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
@click.option("-p", "--predictions-only", "predictions_only", default=False, type=bool, is_flag=True)
def main(config_path: str, predictions_only: bool = False):
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        config_path (str): path to the config file for the training experiment.
        predictions_only (bool): flag to indicate if only the predictions should be saved and not analyzed further.
    """
    config = load_config(config_path)
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = "Paul/hatecheck"
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device")
    padding = config["processing"]["padding"]
    hate_speech_label = config["task"]["hate_speech_label"]
    benign_label = config["task"]["benign_label"]
    predictions_file = config["task"].get("predictions", None)

    accelerator = Accelerator(cpu=device == "cpu")
    device = accelerator.device

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
    )
    additional_information = datasets.load_dataset(
        "csv",
        data_files="data/hatecheck/test_suite_cases_additional.csv",
        sep=";"
    )["train"]

    class_to_label_converter = ClassToLabel(hate_speech_label, benign_label)
    dataset = dataset["test"].map(class_to_label_converter.convert)

    batch_size = config["pipeline"]["batch_size"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    print(model.bert.embeddings.word_embeddings.weight[0][0])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model_state_dict = {k.replace("module.", ""): v for (k, v) in checkpoint["model"].items()}
    model.load_state_dict(model_state_dict, strict=False)
    metric = evaluate.load("accuracy")
    print(model.bert.embeddings.word_embeddings.weight[0][0])

    logger.info(f" Device used: {device}.")
    logger.info(" Starting evaluating model on the data.")

    if not predictions_file:
        dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)
        model, dataloader = accelerator.prepare(model, dataloader)
        results, predictions, references = evaluate_data(model, dataloader, metric, device)
    else:
        with open(predictions_file, "r") as f:
            content = json.load(f)
        results = content["results"]
        predictions = content["predictions"]
        references = content["references"]

    if predictions_only:
        info_to_save = {
            "results": results,
            "predictions": predictions.int().tolist(),
            "references": references.int().tolist(),
        }

        p = Path(config["pipeline"]["output_path"])
        p.mkdir(exist_ok=True, parents=True)

        with open(config["pipeline"]["output_path"], "w") as f:
            json.dump(info_to_save, f, indent=2)

        quit()

    logger.info(f"Overall Loss: {results['loss']}, Overall Accuracy: {results['accuracy']}")
    plot_confusion_matrix("Overall HateCheck", results['confusion_matrix'])
    results["confusion_matrix"] = results["confusion_matrix"].tolist()

    dataset = dataset.add_column("references", references.int().tolist())
    dataset = dataset.add_column("predictions", predictions.int().tolist())

    filter_evaluator = HSTypeFilters(additional_information)
    filter_evaluator.results["overall"] = {
        "results": results,
        "predictions": predictions.int().tolist(),
        "references": references.int().tolist(),
    }
    filter_evaluator.evaluate(dataset)

    p = Path(config["pipeline"]["output_path"])
    p.mkdir(exist_ok=True)

    with open(config["pipeline"]["output_path"], "w") as f:
        json.dump(filter_evaluator.results, f, indent=2)

    # aggregration strategies:
    # - aggregate per target_type
    # - aggregate per functionality
    # - aggregate per functionality x target_type

    # OLD code:
    # functionalities = list(set(dataset["functionality"]))
    # functionality_to_results = {}
    #
    #
    # for functionality in functionalities:
    #     logger.info(f"Evaluating functionality: {functionality}")
    #
    #     subdataset = dataset.filter(lambda example: example["functionality"] == functionality)
    #     dataloader = get_dataloader(subdataset, tokenizer, batch_size, padding)
    #
    #     results, predictions, references = evaluate_data(model, dataloader, metric, device)
    #
    #     logger.info(f"Loss: {results['loss']}, Accuracy: {results['accuracy']}")
    #     plot_confusion_matrix(functionality, results['confusion_matrix'])
    #
    #     correct_subdataset = subdataset.filter(
    #         lambda example,
    #         idx: references[idx] == predictions[idx],
    #         with_indices=True,
    #     )
    #     incorrect_subdataset = subdataset.filter(
    #         lambda example,
    #         idx: references[idx] != predictions[idx],
    #         with_indices=True,
    #     )
    #     results["confusion_matrix"] = results["confusion_matrix"].tolist()
    #     functionality_to_results[functionality] = {
    #         "results": results,
    #         "predictions": predictions.int().tolist(),
    #         "references": references.int().tolist(),
    #         "correct": correct_subdataset["test_case"],
    #         "incorrect": incorrect_subdataset["test_case"],
    #     }
    #
    # with open(config["pipeline"]["output_path"], "w") as f:
    #     json.dump(functionality_to_results, f, indent=2)


if __name__ == "__main__":
    main()
