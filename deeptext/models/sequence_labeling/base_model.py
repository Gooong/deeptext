import os
import logging
import dill

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn

from scipy.stats import entropy

from deeptext.utils.graph import freeze_graph, load_graph
from deeptext.utils.csv import read_csv
from deeptext.utils.serialization import save, restore

from utils import read_data

PARAM_KEY_TOKEN_VOCAB_SIZE = "token_vocab_size"
PARAM_KEY_LABEL_VOCAB_SIZE = "label_vocab_size"
PARAM_KEY_MAX_DOCUMENT_LEN = "max_document_len"
PARAM_KEY_EMBEDDING_SIZE = "embedding_size"
PARAM_KEY_DROPOUT_PROB = "dropout_prob"
PARAM_KEY_MODEL_DIR = "model_dir"

FILENAME_TOKEN_VOCAB = "token.vocab"
FILENAME_LABEL_VOCAB = "label.vocab"

TENSOR_NAME_TOKENS      = "deeptext/models/sequence_labeling/tokens"
TENSOR_NAME_LABELS      = "deeptext/models/sequence_labeling/labels"
TENSOR_NAME_PREDICTION  = "deeptext/models/sequence_labeling/prediction"
TENSOR_NAME_LOSS        = "deeptext/models/sequence_labeling/loss"
TENSOR_NAME_LOGITS      = "deeptext/models/sequence_labeling/logits"
TENSOR_NAME_DEBUG1      = "deeptext/models/sequence_labeling/debug1"
TENSOR_NAME_DEBUG2      = "deeptext/models/sequence_labeling/debug2"
TENSOR_NAME_DEBUG3      = "deeptext/models/sequence_labeling/debug3"
TENSOR_NAME_DEBUG4      = "deeptext/models/sequence_labeling/debug4"
TENSOR_NAME_DEBUG5      = "deeptext/models/sequence_labeling/debug5"

class BaseModel(learn.Estimator):

    def __init__(self, params):
        self.params = params

        tf.logging.set_verbosity(tf.logging.INFO)

    def preprocess_fit_transform(self, training_data_path):

        def tokenizer(iterator):
            for value in iterator:
                yield value 

        tokens, labels = read_data(training_data_path)

        self.token_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
        self.token_vocab.fit(tokens)
        self.token_ids = self.preprocess_token_transform(tokens)
        self.params[PARAM_KEY_TOKEN_VOCAB_SIZE] = len(self.token_vocab.vocabulary_)

        self.label_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
        self.label_vocab.fit(labels)
        self.label_ids = self.preprocess_label_transform(labels)
        self.params[PARAM_KEY_LABEL_VOCAB_SIZE] = len(self.label_vocab.vocabulary_)

        model_dir = self.params[PARAM_KEY_MODEL_DIR]

        token_vocab_path = os.path.join(model_dir, FILENAME_TOKEN_VOCAB)
        save(self.token_vocab, token_vocab_path)

        label_vocab_path = os.path.join(model_dir, FILENAME_LABEL_VOCAB)
        save(self.label_vocab, label_vocab_path)

    def preprocess_token_transform(self, tokens):
        token_ids = self.token_vocab.transform(tokens)
        return np.array(list(token_ids))

    def preprocess_label_transform(self, labels):
        label_ids = self.label_vocab.transform(labels)
        return np.array(list(label_ids))

    def build_model(self):

        TOKEN_VOCAB_SIZE = self.params[PARAM_KEY_TOKEN_VOCAB_SIZE]
        LABEL_VOCAB_SIZE = self.params[PARAM_KEY_LABEL_VOCAB_SIZE]
        MAX_DOCUMENT_LEN = self.params[PARAM_KEY_MAX_DOCUMENT_LEN]
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
            logits = tf.reshape(logits, [-1, MAX_DOCUMENT_LEN, LABEL_VOCAB_SIZE], name=TENSOR_NAME_LOGITS)

            loss = None
            train_op = None
            if y is not None:
                zeros_with_shape = tf.zeros_like(y, dtype=tf.int64)
                weights = tf.to_double(tf.reshape(tf.not_equal(zeros_with_shape, y), [-1]))

                target = tf.one_hot(y, LABEL_VOCAB_SIZE, 1, 0)
                loss = tf.contrib.losses.softmax_cross_entropy(
                        tf.reshape(logits, [-1, LABEL_VOCAB_SIZE]),
                        tf.reshape(target, [-1, LABEL_VOCAB_SIZE]),
                        weights=weights)
                named_loss = tf.identity(loss, name=TENSOR_NAME_LOSS)
                
                # Create a training op.
                train_op = tf.contrib.layers.optimize_loss(
                    loss,
                    tf.contrib.framework.get_global_step(),
                    optimizer='Adam',
                    learning_rate=0.01)
            
            return ({
                'class': tf.argmax(logits, 2, name=TENSOR_NAME_PREDICTION),
                'prob': tf.nn.softmax(logits)
            }, named_loss, train_op)
        
        self.model = learn.Estimator(
                model_fn=model_fn, 
                model_dir=self.params[PARAM_KEY_MODEL_DIR], 
                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

    def fit(self, steps, batch_size=None, validation_token_path=None, validation_label_path=None):
        def input_fn():
            return tf.placeholder_with_default(self.token_ids, name=TENSOR_NAME_TOKENS, shape=[None, self.params[PARAM_KEY_MAX_DOCUMENT_LEN]]), \
                   tf.placeholder_with_default(self.label_ids, name=TENSOR_NAME_LABELS, shape=[None, self.params[PARAM_KEY_MAX_DOCUMENT_LEN]])

        monitors = []
        if validation_token_path is not None and validation_label_path is not None:
            tokens = read_csv(validation_token_path)
            token_ids = self.preprocess_token_transform(tokens)
    
            labels = read_csv(validation_label_path)
            label_ids = self.preprocess_label_transform(labels)

            validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                    token_ids,
                    label_ids,
                    every_n_steps=10)
            monitors.append(validation_monitor)

        self.model.fit(input_fn=input_fn, steps=steps, batch_size=batch_size, monitors=monitors)

    def frozen_save(self):
        model_dir = self.params[PARAM_KEY_MODEL_DIR]
        freeze_graph(model_dir, [TENSOR_NAME_PREDICTION, TENSOR_NAME_LOSS, TENSOR_NAME_LOGITS])

    def predict(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        labels_transform = self.sess.run(self.prediction_tensor, feed_dict={
            self.tokens_tensor: tokens_transform
            })

        labels = []
        for item in self.label_vocab.reverse(labels_transform):
            labels.append(item.split(' '))
        return labels

    def logits(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        logits = self.sess.run(self.logits_tensor, feed_dict={
            self.tokens_tensor: tokens_transform
            })
        entropy_list = []
        for i in xrange(len(tokens)):
            max_entropy = 0
            for j in xrange(len(tokens[i])):
                cur_entropy = entropy(logits[i][j])
                max_entropy = max(cur_entropy, max_entropy)
            entropy_list.append(max_entropy)
        return entropy_list

    def evaluate(self, testing_token_path, testing_label_path):
        tokens = read_csv(testing_token_path)
        labels = read_csv(testing_label_path)

        predicted_labels = self.predict(tokens)

        corre_cnt = 0
        total_cnt = 0
        for i in xrange(len(labels)):
            total_cnt += len(labels[i])
            for j in xrange(min(len(labels[i]), len(predicted_labels[i]))):
                if labels[i][j] == predicted_labels[i][j]:
                    corre_cnt += 1
        logging.info("label accuracy: %.2f", 1.0 * corre_cnt / total_cnt)

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

        model.frozen_graph = load_graph(model_dir)
        model.tokens_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_TOKENS + ':0')
        model.labels_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_LABELS + ':0')
        model.prediction_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_PREDICTION + ':0')
        model.logits_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_LOGITS + ':0')
        model.loss_tensor = model.frozen_graph.get_tensor_by_name('prefix/' + TENSOR_NAME_LOSS + ':0')
        model.sess = tf.Session(graph=model.frozen_graph)

        return model
