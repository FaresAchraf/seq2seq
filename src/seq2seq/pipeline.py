from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', 'src'))
sys.path.append(CODE_DIR)

from sklearn.pipeline import Pipeline

from seq2seq.processing import  preprocessors as pp

def create_pipeline(encoder_model, decoder_model, input_tokenizer, output_tokenizer) -> Pipeline:
    """

    :param model: pretrained Bert models
    :param tokenizer: pretrained tokenizer
    :return: Pipeline for prediction or training
    """
    classifier_pipe = Pipeline([
                            ("RemovePunct", pp.RemovePunct()),
                            ("RM_ASCII", pp.RmAscii()),
                            ("RmDigits_Lower", pp.RmDigitsLower()),
                            ("Tokenizer", pp.Tokenizer(input_tokenizer, output_tokenizer)),
                            ("Model", pp.Prediction_Model(encoder_model,decoder_model))
    ])
    return classifier_pipe

def create_train_pipeline(model, input_tokenizer, output_tokenizer) -> Pipeline:
    """

    :param model: pretrained Bert models
    :param tokenizer: pretrained tokenizer
    :return: Pipeline for prediction or training
    """
    classifier_pipe = Pipeline([
                            ("RemovePunct", pp.RemovePunct()),
                            ("RM_ASCII", pp.RmAscii()),
                            ("RmDigits_Lower", pp.RmDigitsLower()),
                            ("Tokenizer", pp.TrainTokenizer(input_tokenizer, output_tokenizer)),
                            ("Model", pp.TrainingModel(model))
    ])
    return classifier_pipe




from transformers import pipeline
