import sys
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory"
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../..', "seq2seq"))
sys.path.append(CODE_DIR)
import pandas as pd
import joblib
from config import config
from sklearn.pipeline import Pipeline
from transformers import TFBertForSequenceClassification, BertTokenizer
import warnings
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tokenizers import Tokenizer
import json
import models.build_training_model as bt
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

warnings.filterwarnings('ignore', message='foo bar')


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = "regression_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline


def build_embedding_matrix(word2idx_inputs):
    # store all the pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    with open(config.EMBEDDING_DIR) as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))
    num_words = min(config.MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, config.EMBEDDING_DIM))
    for word, i in word2idx_inputs.items():
        if i < config.MAX_NUM_WORDS:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words


def load_prediction_model(max_len_target=3):
    input_tokenizer = Tokenizer.from_file(config.INPUT_TOKENIZER_DIR)
    embedding_matrix, num_words = build_embedding_matrix(input_tokenizer.get_vocab())
    with open(config.OUTPUt_TOKENIZER_DIR) as f:
        data = json.load(f)
        output_tokenizer = tokenizer_from_json(data)
    model_class = bt.BuildModels(embedding_matrix,
                                 EMBEDDING_DIM=config.EMBEDDING_DIM,
                                 input_length=25,
                                 number_words=num_words,
                                 max_len_target=max_len_target,
                                 number_words_output=len(output_tokenizer.word_index) + 1)
    encoder_model, decoder_model = model_class.build_prediction_model()
    encoder_model.load_weights(config.WEIGHTS_DIR, by_name=True)
    decoder_model.load_weights(config.WEIGHTS_DIR, by_name=True)

    return encoder_model, decoder_model, input_tokenizer, output_tokenizer


def load_training_model(df, max_len_target=3):
    input_tokenizer = Tokenizer.from_file(config.INPUT_TOKENIZER_DIR)
    embedding_matrix, num_words = build_embedding_matrix(input_tokenizer.get_vocab())
    with open(config.OUTPUt_TOKENIZER_DIR) as f:
        data = json.load(f)
        output_tokenizer = tokenizer_from_json(data)
    output_tokenizer.fit_on_texts(list(df['target_texts_inputs']) + list(df['target_texts']))
    with open(config.OUTPUt_TOKENIZER_DIR) as f:
        data = json.load(f)
        output_tokenizer2 = tokenizer_from_json(data)
    if len(output_tokenizer.word_index) == len(output_tokenizer2.word_index):
        print("yes")
        with open(config.OUTPUt_TOKENIZER_DIR) as f:
            data = json.load(f)
            output_tokenizer = tokenizer_from_json(data)

    model_class = bt.BuildModels(embedding_matrix,
                                 EMBEDDING_DIM=config.EMBEDDING_DIM,
                                 input_length=25,
                                 number_words=num_words,
                                 max_len_target=max_len_target,
                                 number_words_output=len(output_tokenizer.word_index) + 1)
    training_model = model_class.build_training_model()
    training_model.load_weights(config.WEIGHTS_DIR, by_name=True)

    return training_model, input_tokenizer, output_tokenizer


def decode_sequence(encoder_model, decoder_model, input_seq, word2idx_outputs, idx2word_trans, max_len_target=3):
    # Encode the input as state vectors.
    enc_out = encoder_model.predict(input_seq)
    prob = []

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # [s, c] will be updated in each loop iteration
    s = np.zeros((1, config.LATENT_DIM_DECODER))
    c = np.zeros((1, config.LATENT_DIM_DECODER))

    # Create the translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        # Get next word
        idx = np.argmax(o.flatten())

        # End sentence of EOS
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)
            prob.append(o.flatten()[idx])

        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx
    return {"predicted pclass": output_sentence[0], "predicted class": output_sentence[1], "Score pclass": prob[0],
            "Score class": prob[1]}


def load_train_data() -> pd.DataFrame:
    """
    This function load the pretrained classes  and update them using the new training dataset
    :return: a dataset (pandas DataFrame) ,  dictionary {class:id} , id2label: dictionary (id:class)
    """
    df = pd.read_csv(config.TEST_PATH)
    df['cname'] = df['cname'].apply(lambda x: x.replace(' ', "_"))
    df['pcname'] = df['pcname'].apply(lambda x: x.replace(' ', "_"))
    df['output_data'] = df['pcname'] + " " + df['cname']
    df['target_texts'] = df['output_data'] + " <eos>"
    df['target_texts_inputs'] = "<sos> " + df['output_data']
    return df


def load_prediction_data() -> pd.DataFrame:
    """

    :return: the predictionData as pandas DataFrame
    """

    return pd.read_csv(config.TEST_PATH)


def custom_loss(y_true, y_pred):
    # both are of shape N x T x K
    mask = K.cast(y_true > 0, dtype='float32')
    out = mask * y_true * K.log(y_pred)
    return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
    # both are of shape N x T x K
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    # 0 is padding, don't include those
    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total


def new_acc(y_true, y_pred):
    # both are of shape N x T x K
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')
    pcname_mask = tf.expand_dims(K.cast(K.greater(correct[:, 0], 0), dtype='float'), axis=-1)
    correct = correct * pcname_mask
    # 0 is padding, don't include those
    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total
