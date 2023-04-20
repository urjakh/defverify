from pathlib import Path

from datasets import DatasetDict, load_dataset


class HFDatasetCreator:
    """
        TBD.
    """
    def __init__(self, dataset_name: str, dataset_file: str, train_split: float, val_split: float, test_split: float):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.dataset = None

    def load_data(self):
        suffix = Path(self.dataset_file).suffix[1:]
        self.dataset = load_dataset("csv", data_files=self.dataset_file)

    def split_dataset(self):
        if self.dataset_name == "davidson":
            test_val_split = self.val_split + self.test_split
            train_testvalid = self.dataset["train"].train_test_split(test_size=test_val_split)




# 90% train, 10% test + validation
train_testvalid = dataset["train"].train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})