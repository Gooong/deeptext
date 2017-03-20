import os
import dill

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn

from deeptext.models.sequence_labeling.utils import freeze_graph
from deeptext.models.sequence_labeling.utils import load_graph
from deeptext.utils.csv import read_csv
from deeptext.utils.serialization import save, restore

PARAM_KEY_TOKEN_VOCAB_SIZE = "token_vocab_size"
PARAM_KEY_LABEL_VOCAB_SIZE = "label_vocab_size"
PARAM_KEY_MAX_DOCUMENT_LEN = "max_document_len"
PARAM_KEY_EMBEDDING_SIZE = "embedding_size"
PARAM_KEY_DROPOUT_PROB = "dropout_prob"
PARAM_KEY_MODEL_DIR = "model_dir"

FILENAME_TOKEN_VOCAB = "token.vocab"
FILENAME_LABEL_VOCAB = "label.vocab"
FILENAME_FROZEN_GRAPH = "frozen_model.pb"

TENSOR_NAME_TOKENS = "deeptext/models/sequence_labeling/tokens"
TENSOR_NAME_LABELS = "deeptext/models/sequence_labeling/labels"

class BaseModel(learn.Estimator):

    def __init__(self, params):
        self.params = params

    def preprocess_fit_transform(self, training_token_path, training_label_path):

        def tokenizer(iterator):
            for value in iterator:
                yield value 

        tokens = read_csv(training_token_path)

        self.token_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
        self.token_vocab.fit(tokens)
        self.token_ids = self.preprocess_token_transform(tokens)
        self.params[PARAM_KEY_TOKEN_VOCAB_SIZE] = len(self.token_vocab.vocabulary_)

        labels = read_csv(training_label_path)

        self.label_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
        self.label_vocab.fit(labels)
        self.label_ids = self.preprocess_label_transform(labels)
        self.params[PARAM_KEY_LABEL_VOCAB_SIZE] = len(self.label_vocab.vocabulary_)

    def preprocess_token_transform(self, tokens):
        token_ids = self.token_vocab.transform(tokens)
        return np.array(list(token_ids))

    def preprocess_label_transform(self, labels):
        label_ids = self.label_vocab.transform(labels)
        return np.array(list(label_ids))

    def build_model(self, model_dir):
        self.params[PARAM_KEY_MODEL_DIR] = model_dir

        TOKEN_VOCAB_SIZE = self.params[PARAM_KEY_TOKEN_VOCAB_SIZE]
        LABEL_VOCAB_SIZE = self.params[PARAM_KEY_LABEL_VOCAB_SIZE]
        EMBEDDING_SIZE = self.params[PARAM_KEY_EMBEDDING_SIZE]
        DROPOUT_PROB = self.params[PARAM_KEY_DROPOUT_PROB]

        def model_fn(x, y):
       
            word_vectors = tf.contrib.layers.embed_sequence(
                x, vocab_size=TOKEN_VOCAB_SIZE, embed_dim=EMBEDDING_SIZE, scope='words')
            
            cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT_PROB)
            
            output, _ = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32)
            output = tf.reshape(output, [-1, EMBEDDING_SIZE])
            logits = tf.contrib.layers.fully_connected(output, LABEL_VOCAB_SIZE)

            loss = None
            train_op = None
            if y is not None:
                target = tf.one_hot(y, LABEL_VOCAB_SIZE, 1, 0)
                loss = tf.contrib.losses.softmax_cross_entropy(
                        tf.reshape(logits, [-1, LABEL_VOCAB_SIZE]),
                        tf.reshape(target, [-1, LABEL_VOCAB_SIZE]))
                
                # Create a training op.
                train_op = tf.contrib.layers.optimize_loss(
                    loss,
                    tf.contrib.framework.get_global_step(),
                    optimizer='Adam',
                    learning_rate=0.01)
            
            return ({
                'class': tf.argmax(logits, 1, name='deeptext/models/sequence_labeling/labels'),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)
        
        self.model = learn.Estimator(model_fn=model_fn, model_dir=model_dir)

    def fit(self, steps):
        def input_fn():
            return tf.placeholder_with_default(self.token_ids, name=TENSOR_NAME_TOKENS, shape=[None, self.params[PARAM_KEY_MAX_DOCUMENT_LEN]]), \
                   tf.placeholder_with_default(self.label_ids, shape=[None, self.params[PARAM_KEY_MAX_DOCUMENT_LEN]])
        self.model.fit(input_fn=input_fn, steps=steps)

    def frozen_save(self):
        model_dir = self.params[PARAM_KEY_MODEL_DIR]

        token_vocab_path = os.path.join(model_dir, FILENAME_TOKEN_VOCAB)
        save(self.token_vocab, token_vocab_path)

        label_vocab_path = os.path.join(model_dir, FILENAME_LABEL_VOCAB)
        save(self.label_vocab, label_vocab_path)

        freeze_graph(model_dir)

    def predict(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        labels = self.sess.run(self.labels_tensor, feed_dict={
            self.tokens_tensor: tokens_transform
            })
        return labels

    @classmethod
    def restore(cls, model_dir):

        params = {}
        params[PARAM_KEY_MODEL_DIR] = model_dir

        model = cls(params)

        model.token_vocab_path = os.path.join(model_dir, FILENAME_TOKEN_VOCAB)
        assert os.path.exists(model.token_vocab_path), "missing " + FILENAME_TOKEN_VOCAB
        model.token_vocab = restore(model.token_vocab_path)

        model.label_vocab_path = os.path.join(model_dir, FILENAME_LABEL_VOCAB)
        assert os.path.exists(model.label_vocab_path), "missing " + FILENAME_LABEL_VOCAB
        model.label_vocab = restore(model.label_vocab_path)

        model.frozen_graph_path = os.path.join(model_dir, FILENAME_FROZEN_GRAPH)
        assert os.path.exists(model.frozen_graph_path), "missing " + FILENAME_FROZEN_GRAPH
        model.frozen_graph = load_graph(model.frozen_graph_path)
        model.tokens_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_TOKENS + ':0')
        model.labels_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_LABELS + ':0')
        model.sess = tf.Session(graph=model.frozen_graph)

        return model
