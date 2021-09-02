import os, sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Softmax
from tensorflow.keras.preprocessing.text import Tokenizer as Tokenizer2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import sys
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory"
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)
import pandas as pd
from config import config
import joblib


class BuildModels:
    def __init__(self, embedding_matrix, EMBEDDING_DIM, input_length, number_words, max_len_target,number_words_output):
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.input_length = input_length
        self.num_words_output=number_words_output
        self.num_words = number_words
        self.max_len_target = max_len_target
        self.embedding_layer = Embedding(
            self.num_words,
            self.EMBEDDING_DIM,
            input_length=self.input_length,
            weights=[self.embedding_matrix],
            trainable=True,
            name="input_embedding"
        )
        self.encoder_inputs_placeholder = Input(shape=(self.input_length), name="encoder_inputs_placeholder")
        self.encoder = Bidirectional(LSTM(
            config.LATENT_DIM,
            return_sequences=True,
            dropout=0.2,
            name="encoder_lstm"
        ))
        self.decoder_inputs_placeholder = Input(shape=(self.max_len_target,), name="decoder_inputs_placeholder")
        self.decoder_embedding = Embedding(self.num_words_output, EMBEDDING_DIM, trainable=True, name="decoder_embedding")
        self.attn_repeat_layer = RepeatVector(self.input_length, name="attn_repeat_layer")
        self.attn_concat_layer = Concatenate(axis=-1, name="attn_concat_layer")
        self.attn_dense1 = Dense(128, activation='tanh', name="attn_dense1")
        self.attn_dense2 = Dense(1, activation=Softmax(axis=1), name="attn_dense2")
        self.attn_dot = Dot(axes=1, name="attn_dot")
        self.decoder_lstm = LSTM(config.LATENT_DIM_DECODER, return_state=True, name="decoder_lstm")
        self.decoder_dense = Dense(self.num_words_output, activation='softmax', name="decoder_dense")
        self.initial_s = Input(shape=(config.LATENT_DIM_DECODER,), name='s0')
        self.initial_c = Input(shape=(config.LATENT_DIM_DECODER,), name='c0')
        self.context_last_word_concat_layer = Concatenate(axis=2, name="context_last_word_concat_layer")
        self.encoder_outputs_as_input = Input(shape=(25, config.LATENT_DIM * 2,), name="encoder_outputs_as_input")
        self.decoder_inputs_single = Input(shape=(1,), name="decoder_inputs_single")

    def stack_and_transpose(self, x):
        # x is a list of length T, each element is a batch_size x output_vocab_size tensor
        x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
        x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch_size x T x output_vocab_size
        return x

    def one_step_attention(self, h, st_1):
        # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
        # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

        # copy s(t-1) Tx times
        # now shape = (Tx, LATENT_DIM_DECODER)
        st_1 = self.attn_repeat_layer(st_1)

        # Concatenate all h(t)'s with s(t-1)
        # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
        x = self.attn_concat_layer([h, st_1])

        # Neural net first layer
        x = self.attn_dense1(x)

        # Neural net second layer with special softmax over time
        alphas = self.attn_dense2(x)

        # "Dot" the alphas and the h's
        # Remember a.dot(b) = sum over a[t] * b[t]
        context = self.attn_dot([alphas, h])

        return context

    def build_training_model(self):
        x = self.embedding_layer(self.encoder_inputs_placeholder)
        encoder_outputs = self.encoder(x)
        decoder_inputs_x = self.decoder_embedding(self.decoder_inputs_placeholder)
        s = self.initial_s
        c = self.initial_c
        # collect outputs in a list at first
        outputs = []
        for t in range(self.max_len_target):  # Ty times
            # get the context using attention
            context = self.one_step_attention(encoder_outputs, s)

            # we need a different layer for each time step
            selector = Lambda(lambda x: x[:, t:t + 1])
            xt = selector(decoder_inputs_x)

            # combine
            decoder_lstm_input = self.context_last_word_concat_layer([context, xt])

            # pass the combined [context, last word] into the LSTM
            # along with [s, c]
            # get the new [s, c] and output
            o, s, c = self.decoder_lstm(decoder_lstm_input, initial_state=[s, c])

            # final dense layer to get next word prediction
            decoder_outputs = self.decoder_dense(o)
            outputs.append(decoder_outputs)
        # make it a layer
        stacker = Lambda(self.stack_and_transpose)
        outputs = stacker(outputs)
        # create the model
        model = Model(
            inputs=[
                self.encoder_inputs_placeholder,
                self.decoder_inputs_placeholder,
                self.initial_s,
                self.initial_c,
            ],
            outputs=outputs
        )

        return model

    def build_prediction_model(self):
        decoder_inputs_single_x = self.decoder_embedding(self.decoder_inputs_single)
        # no need to loop over attention steps this time because there is only one step
        context = self.one_step_attention(self.encoder_outputs_as_input, self.initial_s)
        decoder_lstm_input = self.context_last_word_concat_layer([context, decoder_inputs_single_x])
        o, s, c = self.decoder_lstm(decoder_lstm_input, initial_state=[self.initial_s, self.initial_c])
        decoder_outputs = self.decoder_dense(o)
        decoder_model = Model(
            inputs=[
                self.decoder_inputs_single,
                self.encoder_outputs_as_input,
                self.initial_s,
                self.initial_c
            ],
            outputs=[decoder_outputs, s, c]
        )
        x = self.embedding_layer(self.encoder_inputs_placeholder)
        encoder_outputs = self.encoder(x)
        encoder_model = Model(self.encoder_inputs_placeholder, encoder_outputs)

        return encoder_model, decoder_model
