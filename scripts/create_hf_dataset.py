import ast
from pathlib import Path
from typing import List

import click
from datasets import DatasetDict, load_dataset

DATASETS = ["davidson", "founta", "kennedy", "mathew", "talat_hovy", "vidgen"]


class HFDatasetCreator:
    """
        TBD.
    """
    def __init__(self, dataset_name: str, dataset_file: str, dataset_split: List[float], seed: int = 0):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.train_split = dataset_split[0]
        self.val_split = dataset_split[1]
        self.test_split = dataset_split[2]
        self.seed = seed

        assert self.train_split + self.val_split + self.test_split == 1, "The split distribution does not sum up to 1."

        self.dataset = None

    def load_data(self):
        suffix = Path(self.dataset_file).suffix[1:]
        self.dataset = load_dataset(suffix, data_files=self.dataset_file)

    def split_dataset(self):
        if self.dataset_name == "davidson" or self.dataset_name == "founta":
            test_val_split = self.val_split + self.test_split
            train_testvalid = self.dataset["train"].train_test_split(test_size=test_val_split, seed=self.seed)

            left_over_split = 1 - self.train_split
            new_test_split = self.test_split / left_over_split

            valid_test = train_testvalid["test"].train_test_split(test_size=new_test_split, seed=self.seed)

            self.dataset = DatasetDict({
                "train": train_testvalid["train"],
                "val": valid_test["train"],
                "test": valid_test["test"],
            })

    def save_dataset(self, output: str):
        self.dataset.save_to_disk(output)


@click.command()
@click.option("-n", "--dataset-name", "dataset_name", required=True, type=str)
@click.option("-p", "--path", "path", required=True, type=str)
@click.option("-o", "--output", "output", required=True, type=str)
@click.option("-s", "--dataset-split", "dataset_split", type=str)
def main(dataset_name: str, path: str, output: str, dataset_split: str):
    assert dataset_name in DATASETS, f"This dataset is not supported, please provide one of the following datasets: " \
                                     f"{DATASETS} "

    dataset_split = ast.literal_eval(dataset_split)
    if dataset_name == "davidson":
        davidson_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        davidson_creator.load_data()
        davidson_creator.split_dataset()
        davidson_creator.save_dataset(output)
    elif dataset_name == "founta":
        davidson_creator = HFDatasetCreator(dataset_name, path, dataset_split)
        davidson_creator.load_data()
        davidson_creator.split_dataset()


if __name__ == "__main__":
    main()