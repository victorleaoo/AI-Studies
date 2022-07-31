# Imports

from fastbook import *

from fastai.basics import *
from fastai.text.core import *
from fastai.text.data import *
from fastai.text.models.core import *
from fastai.text.models.awdlstm import *
from fastai.callback.rnn import *
from fastai.callback.progress import *
from fastai.vision import *
from fastai.text.all import *
from fastai.text import *

import pandas as pd
import argparse

from datetime import datetime

import mlflow

import random
import torch
import numpy as np

# Arguments for the fit_one_cycle
def parse_args():
    parser = argparse.ArgumentParser(description="Fastai Text Classification")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: 5). Note it takes about 1 min per epoch",
    )
    return parser.parse_args()

# Important initializations
def initial_executions():
    seed = 42
    random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

# Get train and test datasets -> drop all columns but text_tags and class
def get_data():
    df_train = pd.read_csv('train_tags.csv', delimiter=',',  error_bad_lines=False, index_col=False)
    df_train = df_train.drop(['text'], axis=1)
    
    df_test = pd.read_csv('test_tags.csv', delimiter=',', error_bad_lines=False)
    df_test = df_test.drop(['text'], axis=1)

    return df_train, df_test

# Get Features
def get_features(r):
    return r['text']

# Get Labels
def get_labels(r):
    return r['class']

# Create and return Train DataBlock
def create_datablock_train(df_train):
    dblock_train = DataBlock(
            blocks=(TextBlock.from_df(text_cols=['text_tags', 'class'], seq_len=72), CategoryBlock),
            get_x=get_features, 
            get_y=get_labels)

    dloader_train = dblock_train.dataloaders(df_train, bs=128)
    return dloader_train

# Create and return Test DataBlock
def create_datablock_test(df_test):
    dblock_test_tags = DataBlock(
    blocks=(TextBlock.from_df('text_tags', seq_len=72), CategoryBlock),
            get_x=get_features, 
            get_y=get_labels)

    dloader_test = dblock_test_tags.dataloaders(df_test, bs=64)
    test_dl = dloader_test.test_dl(df_test['text_tags'])
    return test_dl

def main():
    initial_executions()

    # Get datasets
    df_train, df_test = get_data()

    # Arguments
    args = parse_args()

    # Auto logging
    mlflow.fastai.autolog()

    # Get Train DataBlock
    dl_train = create_datablock_train(df_train)

    # Get Test DataBlock
    dl_test = create_datablock_test(df_test)

    # Learner Model
    learn = text_classifier_learner(dl_train, 
                                AWD_LSTM, 
                                drop_mult=0.5, path='\checkpoints', 
                                metrics=[error_rate, accuracy, Perplexity()]).to_fp16()
    
    # Start MLflow session
    with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
        cbs = [SaveModelCallback()]
        learn.fine_tune(args.epochs, args.lr, cbs=cbs)
        
        # Confusion Matrix and Validation
        preds = learn.get_preds(dl=dl_test, with_decoded=True)
        # interp = ClassificationInterpretation.from_learner(learn)
        # confusion_matrix_image = interp.plot_confusion_matrix(figsize=(4,4), dpi=100)
        # confusion_matrix_image.figure.savefig('confusion_matrix_fastai.png')
        learn.validate()

if __name__ == "__main__":
    main()