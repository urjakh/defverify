import click
import pandas as pd
import json

DATASETS = ["talat_hovy", "founta"]


class DatasetReader:
    def __init__(self, dataset_file_path: str, dataset: str):
        self.dataset_file_path = dataset_file_path
        self.dataset_name = dataset

        assert self.dataset_name in DATASETS, "Please provide a valid dataset."

        self.dataset = self.read_dataset()

    def read_dataset(self):
        if self.dataset_name == "talat_hovy":
            return pd.read_csv(self.dataset_file_path, header=None)

    def get_tweet_ids(self):
        if self.dataset_name == "talat_hovy":
            return self.get_tweet_ids_talat_hovy()

    def get_tweet_ids_talat_hovy(self):
        return self.dataset[0].to_list()


def merge(dataset_file_path: str, tweet_text_file_path: str, dataset: str, save_file_path: str):
    dataset_reader = DatasetReader(dataset_file_path, dataset)
    tweet_ids = dataset_reader.get_tweet_ids()

    with open(tweet_text_file_path, "r") as f:
        ids_to_texts = json.load(f)

    texts = [ids_to_texts[str(tweet_id)] for tweet_id in tweet_ids]

    dataset = dataset_reader.dataset
    dataset["text"] = texts
    dataset.columns = ["tweet_id", "label", "text"]
    dataset.to_csv(save_file_path, sep='\t')


@click.command()
@click.option("-d", "--dataset-file-path", "dataset_file_path", required=True, type=str)
@click.option("-t", "--tweet-text-file-path", "tweet_text_file_path", required=True, type=str)
@click.option("-n", "--dataset", "dataset", required=True, type=str)
@click.option("-s", "--save-file-path", "save_file_path", required=True, type=str)
def main(dataset_file_path: str, tweet_text_file_path: str, dataset: str, save_file_path: str):
    """
    Args:
        TBA.
    """
    merge(dataset_file_path, tweet_text_file_path, dataset, save_file_path)


if __name__ == "__main__":
    main()
