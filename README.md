# robustness-albert

This is the code for ["DefVerify: Do Hate Speech Models Reflect Their Dataset's Definition?"](https://arxiv.org/abs/2410.15911), accepted to COLING 2025.

models will be uploaded and shared soon!

## setup. 
To run the training and evaluation for this paper, please set up the environment: 
```bash 
# Create environment.
conda create -n hs-generalization python=3.9
conda activate hs-generalization

# Install packages.
python setup.py develop
pip install -r requirements.txt
```

## converting datasets to huggingface format. 
Before we start training, we need to convert the datasets we are using into the HuggingFace format and save it (if it is not on HuggingFace already). 
We can do this by using the `scripts/create_hf_dataset.py` script. Add code if you are using a new dataset that requires a different conversion.  We can run the script as follows: 
```bash
python scripts/create_hf_dataset.py -n DATASET_NAME -p PATH_TO_DATASETFILE_OR_FOLDER -o PATH_TO_OUTPUT_HUGGINGFACE_FORMAT -s [OPTIONAL] STRING_IN_LIST_FORMAT_INDICATING_SPLIT_PERCENTAGE
```

This file should work out of the box on the original format of the datasets tested in the paper.

## training.
First, create a config file (see `configs/train/example_config.json` for an example). 

Then, run the following:
```bash
hs_generalization/train.py -c configs/CONFIG_FILE_NAME.json
```

## evaluation. 
To evaluate a model on the test set, create a config file (see `configs/test/example_config.json`) and run the following: 
```bash
hs_generalization/test.py -c configs/CONFIG_FILE_NAME.json
```

To evaluate a model on HateCheck, create a config file (see `configs/hatecheck/example_config.json`) and run the following: 
```bash
hs_generalization/evaluate_checklist.py -c configs/CONFIG_FILE_NAME.json
```

To do cross-evaluation, create a config file (see `configs/cross-eval/example_config.json`) and run the following: 
```bash
hs_generalization/cross_evaluation.py -c configs/CONFIG_FILE_NAME.json
```
