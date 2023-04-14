import click
import tweepy
from configparser import ConfigParser
import pandas as pd
from tqdm import tqdm
import json

DATASETS = ["talat_hovy", "founta"]


class TweetRetriever:
    def __init__(self, keys_file_path: str):
        self.keys_file_path = keys_file_path

        self.config = ConfigParser()
        self.config.read(self.keys_file_path)
        self.api = self.setup_api()

    def setup_api(self):
        # TODO: change configparser to dotenv
        auth = tweepy.OAuthHandler(self.config["KEYS"]["CONSUMER_KEY"], self.config["KEYS"]["CONSUMER_SECRET"])
        auth.set_access_token(self.config["KEYS"]["OAUTH_TOKEN"], self.config["KEYS"]["OAUTH_TOKEN_SECRET"])
        return tweepy.API(auth, wait_on_rate_limit=True)

    def get_tweet(self, tweet_id: str):
        return self.api.get_status(tweet_id)

    @staticmethod
    def get_tweet_text(tweet):
        return tweet.text


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


def get_tweets(keys_file_path: str, dataset_file_path: str, dataset: str, save_file_path: str):
    tweet_retriever = TweetRetriever(keys_file_path)
    dataset_reader = DatasetReader(dataset_file_path, dataset)

    tweet_ids = dataset_reader.get_tweet_ids()

    id_to_text = {}
    for tweet_id in tqdm(tweet_ids):
        try:
            tweet = tweet_retriever.get_tweet(tweet_id)
            tweet_text = tweet_retriever.get_tweet_text(tweet)
        except Exception as e:
            tweet_text = f"TWEETRETRIEVER ERROR: {e}"

        id_to_text[tweet_id] = tweet_text

    with open(save_file_path, "w") as f:
        json.dump(id_to_text, f)


@click.command()
@click.option("-k", "--keys-file-path", "keys_file_path", required=True, type=str)
@click.option("-d", "--dataset-file-path", "dataset_file_path", required=True, type=str)
@click.option("-n", "--dataset", "dataset", required=True, type=str)
@click.option("-s", "--save-file-path", "save_file_path", required=True, type=str)
def main(keys_file_path: str, dataset_file_path: str, dataset: str, save_file_path: str):
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        TBA.
    """
    get_tweets(keys_file_path, dataset_file_path, dataset, save_file_path)


if __name__ == "__main__":
    main()






