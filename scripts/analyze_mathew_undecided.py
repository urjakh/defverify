from pathlib import Path

import click as click
import numpy as np
import json
import pandas as pd


def get_annotated_data(params):
    """ FROM ORIGINAL GITHUB"""
    # temp_read = pd.read_pickle(params['data_file'])
    with open(params['data_file'], 'r') as fp:
        data = json.load(fp)
    dict_data = []
    for key in data:
        temp = {}
        temp['post_id'] = key
        temp['text'] = data[key]['post_tokens']
        final_label = []
        for i in range(1, 4):
            temp['annotatorid' + str(i)] = data[key]['annotators'][i - 1]['annotator_id']
            #             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target' + str(i)] = data[key]['annotators'][i - 1]['target']
            temp['label' + str(i)] = data[key]['annotators'][i - 1]['label']
            final_label.append(temp['label' + str(i)])

        final_label_id = max(final_label, key=final_label.count)
        temp['rationales'] = data[key]['rationales']

        if (Path(params['class_names']).name == 'classes_two.npy'):
            if (final_label.count(final_label_id) == 1):
                temp['final_label'] = 'undecided'
            else:
                if (final_label_id in ['hatespeech', 'offensive']):
                    final_label_id = 'toxic'
                else:
                    final_label_id = 'non-toxic'
                temp['final_label'] = final_label_id


        else:
            if (final_label.count(final_label_id) == 1):
                temp['final_label'] = 'undecided'
            else:
                temp['final_label'] = final_label_id

        dict_data.append(temp)
    temp_read = pd.DataFrame(dict_data)
    return temp_read


def analyze(dataset_path, class_names_path, output_path):
    paths = {"data_file": dataset_path, "class_names": class_names_path}
    data = get_annotated_data(paths)

    print(f"Class Distribution: {data['final_label'].value_counts()}")

    undecided_samples = data[data["final_label"] == "undecided"]

    assert Path(output_path).suffix == ".csv", "Please specify the output path as a CSV file."
    undecided_samples.to_csv(output_path)


@click.command()
@click.option("-d", "--dataset_path", "dataset_path", required=True, type=str)
@click.option("-c", "--class_names_path", "class_names_path", required=True, type=str)
@click.option("-o", "--output-predictions", "output_predictions", required=True, type=str)
def main(dataset_path: str, class_names_path: str, output_predictions: str):
    """Function that executes the entire training pipeline.
        TBA
    Args:
        config_path (str): TBA.
        checkpoint_path (str): TBA.
        output_predictions (str): TBA.
    """
    analyze(dataset_path, class_names_path, output_predictions)


if __name__ == "__main__":
    main()

