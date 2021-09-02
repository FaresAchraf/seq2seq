import sys
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

DATA_PATH = abspath(join(THIS_DIR, 'data', 'categorical_data.csv'))


def get_prediction() -> pd.DataFrame:
    """

    :return: Bert Model Class Name Prediction
    """

    df = dm.load_prediction_data()
    encoder_model, decoder_model, input_tokenizer, output_tokenizer = dm.load_prediction_model()
    pp = pipe.create_pipeline(encoder_model, decoder_model, input_tokenizer, output_tokenizer)
    df2 = pp.predict(df)
    return df2


if __name__ == '__main__':
    df2 = get_prediction()

    df2.head()
