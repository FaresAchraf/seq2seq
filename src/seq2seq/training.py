from os.path import dirname, abspath, join
import sys
import warnings
import pandas as pd

warnings.filterwarnings('ignore', message='foo bar')
# Find code directory relative to our directory"
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', 'src'))
sys.path.append(CODE_DIR)

import seq2seq.pipeline as pipe

import seq2seq.processing.data_management as dm



def run_training():
    """

    Function to load and train the pipeline and save the models weights and config in new_model directory
    """
    df = dm.load_train_data()
    training_model, input_tokenizer, output_tokenizer = dm.load_training_model(df)
    print(training_model.summary())
    classifier_pipe = pipe.create_train_pipeline(training_model, input_tokenizer, output_tokenizer)
    print(df.head())
    classifier_pipe.fit(df)
    return df
    #trained_pipe = classifier_pipe.fit(df, df["labels"])
    #trained_pipe['Model'].model.save_pretrained('new_model')


if __name__ == '__main__':
    df = run_training()
