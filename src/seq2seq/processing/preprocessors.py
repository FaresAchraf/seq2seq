import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import pandas as pd
import numpy as np
from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR))
sys.path.append(CODE_DIR)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import data_management as dm
from config import config
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
from datetime import date
import joblib


# Remove punctuation Class
class RemovePunct(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def remove_punct(self, text):
        """

        :param text:
        :return: Text without punctuation
        """
        nopunct = ""
        for c in text:
            if c not in string.punctuation:
                nopunct = nopunct + c
        return nopunct

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """

        :param x:
        :return: apply the remove_puunct to  all the product names
        """
        x['pname'] = x['pname'].apply(self.remove_punct)
        return x


# Remove Non ASCII text Class
class RmAscii(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """

        :param x: input DataFrame
        :return: DataFrame with removed non ASCII characters
        """
        x['pname'] = x['pname'].apply(lambda y: re.sub(r'[^\x00-\x7F]+', "", y))
        return x


## RemoveDigit Class
class RmDigitsLower(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        x['pname'] = x['pname'].apply(lambda y: re.sub('\w*\d\w*', '', y.lower()))
        return x


# Classifier Class
class Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, *, pipe):
        self.pipe = pipe
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self

    def predict(self, X):
        return self.pipe(X)


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, output_tokenizer):
        self.tokenizer = tokenizer
        self.output_tokenizer = output_tokenizer
        self.tokenizer.enable_padding(pad_to_multiple_of=25, length=25)
        self.tokenizer.enable_truncation(max_length=25)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> tuple:
        """

        :param X: pd.DataFrame ( preprocessed data )
        :return: pd.DataFrame ( preprocessed data ) , dict (Tokenized data)
        """
        idx2word_trans = self.output_tokenizer.index_word
        word2idx_outputs = self.output_tokenizer.word_index
        input_sequences = [i.ids for i in self.tokenizer.encode_batch(X['pname'])]
        return input_sequences, word2idx_outputs, idx2word_trans, X


class Prediction_Model(BaseEstimator, TransformerMixin):
    def __init__(self, encoder=None, decoder=None):
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, X, y):
        return self

    def predict(self, X: tuple) -> pd.DataFrame:
        t = X[0]
        result = pd.DataFrame(map(lambda x: dm.decode_sequence(self.encoder,
                                                               self.decoder,
                                                               input_seq=[x],
                                                               word2idx_outputs=dict(X[1]),
                                                               idx2word_trans=dict(X[2])),
                                  t))
        df = pd.concat([result, X[3]], axis=1)
        df['algo_name'] = "Seq2SeqPclassClass"
        t = time.localtime()
        today = date.today()
        pclass2categoryid = joblib.load(config.PCLASS2CATEGORY)
        d1 = today.strftime("%d/%m/%Y")
        current_time = time.strftime("%H:%M", t)
        df['created'] = str(d1) + " " + current_time
        df['algo_version'] = "1.0.0"
        concatenated_classes = df['predicted pclass']+" "+df['predicted class']
        df['category_id'] = concatenated_classes.map(pclass2categoryid)
        df.dropna(subset=['category_id'],inplace=True)
        df['score'] = (df['Score pclass']+df['Score class'])/2

        return df[['category_id', 'score', 'created', 'algo_name', 'algo_version']]
        return df
class TrainTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, output_tokenizer):
        self.tokenizer = tokenizer
        self.output_tokenizer = output_tokenizer
        self.tokenizer.enable_padding(pad_to_multiple_of=25, length=25)
        self.tokenizer.enable_truncation(max_length=25)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> tuple:
        """

        :param X: pd.DataFrame ( preprocessed data )
        :return: pd.DataFrame ( preprocessed data ) , dict (Tokenized data)
        """

        decoder_inputs = np.array(self.output_tokenizer.texts_to_sequences(np.array(X['target_texts'])))
        decoder_targets = self.output_tokenizer.texts_to_sequences(list(X['target_texts_inputs']))
        decoder_targets_one_hot = np.zeros(
            shape=(
                len(decoder_inputs),
                3,
                len(self.output_tokenizer.index_word) + 1
            ),
            dtype="float32"
        )

        for i, sequence in enumerate(decoder_targets):
            for j, word in enumerate(sequence):
                if word != 0:
                    decoder_targets_one_hot[i, j, word] = 1
        input_sequences = np.array([np.array(i.ids) for i in self.tokenizer.encode_batch(X['pname'])])
        return input_sequences, decoder_inputs, decoder_targets_one_hot


class TrainingModel(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        input_sequences = X[0]
        self.model.compile(optimizer=Adam(0.001), loss=dm.custom_loss, metrics=[dm.new_acc])
        z = np.zeros((len(input_sequences), config.LATENT_DIM_DECODER))
        self.model.fit(
            [np.array(input_sequences), X[1], z, z],
            X[2],
            epochs=2,
            batch_size=config.BATCH_SIZE
        )
