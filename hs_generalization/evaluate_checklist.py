import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Any, Dict, Union, List

import click
import datasets
import evaluate
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Metric, Dataset
from evaluate import EvaluationModule
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification

from hs_generalization.train import get_dataloader
from hs_generalization.utils import load_config, get_dataset, plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

dataset_to_dominance = {
    "davidson": "unknown",
    "founta": "unknown",
    "kennedy": "yes",
    "mathew": "yes",
    "vidgen": "no",
    "talat_hovy": "no",
}


placeholder_to_values = {
    "[IDENTITY_S]": ["white person", "man"],
    "[IDENTITY_P]": ["white people", "men"],
    "[IDENTITY_A]": ["male", "white"],
}

values_to_feature = {
    "white person": {
        "target_type": "race",
        "dominance": "yes"
    },
    "man": {
        "target_type": "gender",
        "dominance": "yes"
    },
    "white people": {
        "target_type": "race",
        "dominance": "yes"
    },
    "men": {
        "target_type": "gender",
        "dominance": "yes"
    },
    "male": {
        "target_type": "gender",
        "dominance": "yes"
    },
    "white": {
        "target_type": "race",
        "dominance": "yes"
    },
}


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
) -> Tuple[Dict, torch.Tensor, torch.Tensor, List]:
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
    confidences = torch.tensor([])
    losses = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            references = torch.cat([references, batch["labels"].to("cpu")])
            confidences = torch.cat([confidences, outputs.logits.softmax(dim=-1).to("cpu")])

            losses.append(outputs.loss.detach().cpu().numpy())

    eval_loss = np.mean(losses)
    score = metric.compute(predictions=predictions, references=references)

    cm = confusion_matrix(references, predictions)

    results = {
        "loss": float(eval_loss),
        "confusion_matrix": cm.tolist()
    }

    results = results | score
    return results, predictions.int().tolist(), references.int().tolist(), confidences.tolist()


class HSTypeFilters:
    def __init__(self, added_information: Dataset, to_evaluate: Union[str, List[str]] = "all"):
        if to_evaluate == "all":
            self.to_evaluate = []

        self.to_evaluate = to_evaluate
        if self.to_evaluate == "all":
            self.to_evaluate = [
                "non_and_hate", "functionalities", "target_types", "dominance", "explicit_references", "consequences",
                "group_insult", "in_group"
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
            "explicit_reference": "explicit_ref",
            "group_insult": "group_insult",
            "in_group": "in_group"
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

    def evaluate_group_insult(self, dataset: Dataset):
        logger.info("Evaluating Group Insult")
        for insult in ["yes", "no"]:
            case_ids, info, subdataset = self.split_on_type(
                "group_insult", insult, self.added_information, dataset
            )

            accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])
            logger.info(f"Accuracy Group Insult {insult}: {accuracy}")
            self.results["group_insult"][insult]["overall"] = {
                "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
            }

    def evaluate_in_group(self, dataset: Dataset):
        logger.info("Evaluating In Group")
        case_ids, info, subdataset = self.split_on_type(
            "in_group", "yes", self.added_information, dataset
        )
        accuracy = self.accuracy.compute(predictions=subdataset["predictions"], references=subdataset["references"])
        logger.info(f"Accuracy In Group: {accuracy}")
        self.results["in_group"]["yes"]["overall"] = {
            "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
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
            elif evaluate_type == "group_insult":
                self.evaluate_group_insult(dataset)
            elif evaluate_type == "in_group":
                self.evaluate_in_group(dataset)


def add_dominant_data(dataset, dataset_name, tokenizer, padding):
    templates = (dataset["case_templ"])
    template_to_functionality = dict(zip(dataset["case_templ"], dataset["functionality"]))

    cases = []
    for placeholder, values in placeholder_to_values.items():
        placeholder_templates = set([template for template in templates if placeholder in template])
        for template in placeholder_templates:
            functionality = template_to_functionality[template]
            new_label = "non-hateful"
            if functionality[-2:] == "_h":
                new_label = "hateful"
                if dataset_to_dominance[dataset_name] == "no":
                    new_label = "non-hateful"
            for value in values:

                # + 201 because there's a mismatch between length and case ids.
                case = {
                    "case_id": len(dataset) + len(cases) + 201,
                    "test_case": template.replace(placeholder, value),
                    "labels": new_label,
                    "target_ident": "men"
                }

                if value in ["white person", "white people", "white"]:
                    case["target_ident"] = "white people"

                tokenized = tokenizer(case["test_case"], padding=padding, max_length=512, truncation=True)
                case["input_ids"] = tokenized["input_ids"]
                if "token_type_ids" in tokenizer.model_input_names:
                    case["token_type_ids"] = tokenized["token_type_ids"]
                case["attention_mask"] = tokenized["attention_mask"]

                cases.append(case)

    return cases


def get_racism_sexism_labels(
        dataset: Dataset,
        additional_information: Dataset,
        sexism_label: int = 0,
        racism_label: int = 1,
        benign_label: int = 2
):
    def get_label(example, index):
        label = benign_label
        if example["labels"] == "hateful":
            if additional_information[index]["target_ident"] in ["women", "trans people"]:
                    label = sexism_label
            elif additional_information[index]["target_ident"] in ["black people", "Muslims", "immigrants"]:
                label = racism_label

        example["labels"] = label
        return example

    dataset = dataset.map(get_label, with_indices=True)
    # samples = []
    # for j, sample in enumerate(dataset):
    #     label = benign_label
    #     # After this index the dominant ones are added, always benign since only minorities considered for this dataset.
    #     if j < 3728:
    #         if additional_information[j]["target_ident"] in ["women", "trans people"]:
    #             label = sexism_label
    #         elif additional_information[j]["target_ident"] in ["black people", "Muslims", "immigrants"]:
    #             label = racism_label
    #     sample["labels"] = label
    #     samples.append(sample)

    return dataset


def test_dominance_old(dataset, dataset_name, tokenizer, padding, additional_information):
    cases = add_dominant_data(dataset, dataset_name, tokenizer, padding)
    for case in tqdm(cases):
        dataset = dataset.add_item(case)
    for case in tqdm(cases):
        case_copy = deepcopy(case)
        case_copy["dominance"] = "yes"
        additional_information = additional_information.add_item(case_copy)

    return cases, dataset, additional_information


def test_dominance(
        dataset, dataset_name, tokenizer, padding, batch_size, model, metric, device, class_to_label_converter, benign_label=2
):
    def change_label(example):
        example["labels"] = benign_label
        return example

    cases = add_dominant_data(dataset, dataset_name, tokenizer, padding)
    dataset = Dataset.from_list(cases)
    case_ids = dataset["case_id"]
    target_ident = dataset["target_ident"]
    dataset = dataset.remove_columns(["case_id", "test_case", "target_ident"])

    if dataset_name != "talat_hovy":
        dataset = dataset.map(class_to_label_converter.convert)
    else:
        dataset = dataset.map(change_label)
    dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)
    results, predictions, references, confidences = evaluate_data(model, dataloader, metric, device)

    dataset = dataset.add_column("references", references)
    dataset = dataset.add_column("predictions", predictions)
    dataset = dataset.add_column("case_id", case_ids)
    dataset = dataset.add_column("target_ident", target_ident)
    dataset = dataset.add_column("confidences", confidences)

    logger.info(f"Accuracy Dominant: {results['accuracy']}")

    for group in ["men", "white people"]:
        subdataset = dataset.filter(lambda example: example["target_ident"] == group)

        accuracy = metric.compute(
            predictions=subdataset["predictions"],
            references=subdataset["references"]
        )
        logger.info(f"Accuracy Dominant {group}: {accuracy}")
        results[group] = {
            "case_ids": subdataset["case_id"], "accuracy": accuracy["accuracy"]
        }

    return results, predictions, references, confidences


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
@click.option("-p", "--predictions-only", "predictions_only", default=False, type=bool, is_flag=True)
@click.option("-d", "--test-dominant", "test_dominant", default=False, type=bool, is_flag=True)
def main(config_path: str, predictions_only: bool = False, test_dominant: bool = False):
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        config_path (str): path to the config file for the training experiment.
        predictions_only (bool): flag to indicate if only the predictions should be saved and not analyzed further.
        test_dominant (bool): flag to indicate if dominant groups should be tested separately.
    """
    config = load_config(config_path)
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device")
    padding = config["processing"]["padding"]
    hate_speech_label = config["task"]["hate_speech_label"]
    benign_label = config["task"]["benign_label"]
    predictions_file = config["task"].get("predictions", None)

    if not predictions_file:
        wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"])

    accelerator = Accelerator(cpu=device == "cpu")
    device = accelerator.device

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        "Paul/hatecheck",
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
    )
    additional_information = datasets.load_dataset(
        "csv",
        data_files="data/hatecheck/test_suite_cases_additional_16_06.csv",
        sep=";"
    )["train"]

    dataset = dataset["test"]
    dataset = dataset.filter(lambda example: "spell" not in example["functionality"])
    additional_information = additional_information.filter(lambda example: "spell" not in example["functionality"])
    class_to_label_converter = None
    if dataset_name != "talat_hovy":
        class_to_label_converter = ClassToLabel(hate_speech_label, benign_label)
        dataset = dataset.map(class_to_label_converter.convert)
    else:
        dataset = get_racism_sexism_labels(dataset, additional_information)

    batch_size = config["pipeline"]["batch_size"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model_state_dict = {k.replace("module.", ""): v for (k, v) in checkpoint["model"].items()}
    model.load_state_dict(model_state_dict, strict=False)
    metric = evaluate.load("accuracy")

    logger.info(f" Device used: {device}.")
    logger.info(" Starting evaluating model on the data.")

    if not predictions_file:
        dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)
        model, dataloader = accelerator.prepare(model, dataloader)
        results, predictions, references, confidences = evaluate_data(model, dataloader, metric, device)

        wandb.log(results)
    else:
        with open(predictions_file, "r") as f:
            content = json.load(f)
        results = content["results"]
        predictions = content["predictions"]
        references = content["references"]
        confidences = content["confidences"]

    if predictions_only:
        info_to_save = {
            "results": results,
            "predictions": predictions,
            "references": references,
            "confidences": confidences,
        }

        p = Path(config["pipeline"]["output_predictions"]).parent
        p.mkdir(exist_ok=True, parents=True)

        with open(config["pipeline"]["output_predictions"], "w") as f:
            json.dump(info_to_save, f, indent=2)

        quit()

    logger.info(f"Overall Loss: {results['loss']}, Overall Accuracy: {results['accuracy']}")
    plot_confusion_matrix("Overall HateCheck", results['confusion_matrix'])
    results["confusion_matrix"] = results["confusion_matrix"]

    if test_dominant:
        dominance_results, dominance_predictions, dominance_references, dominance_confidences = test_dominance(
            dataset,
            dataset_name,
            tokenizer,
            padding,
            batch_size,
            model,
            metric,
            device,
            class_to_label_converter
        )

    dataset = datasets.load_dataset("paul/hatecheck")["test"]
    dataset = dataset.filter(lambda example: "spell" not in example["functionality"])

    dataset = dataset.add_column("references", references)
    dataset = dataset.add_column("predictions", predictions)

    filter_evaluator = HSTypeFilters(additional_information)
    filter_evaluator.results["results"] = results,
    filter_evaluator.results["predictions"] = predictions
    filter_evaluator.results["references"] = references
    filter_evaluator.results["confidences"] = confidences
    if test_dominant:
        filter_evaluator.results["dominance"]["results"] = dominance_results
        filter_evaluator.results["dominance"]["predictions"] = dominance_predictions
        filter_evaluator.results["dominance"]["references"] = dominance_references
        filter_evaluator.results["dominance"]["confidences"] = dominance_confidences
    filter_evaluator.evaluate(dataset)

    p = Path(config["pipeline"]["output_predictions"]).parent
    p.mkdir(exist_ok=True, parents=True)

    with open(config["pipeline"]["output_predictions"], "w") as f:
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
