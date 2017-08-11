import os
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn

from scipy.stats import entropy

from deeptext.models.base import Base
import deeptext.utils.serialization
import deeptext.models.constants as constants

from utils import read_data


class SequenceLabeling(Base):
    def __init__(self, params):
        super(SequenceLabeling, self).__init__(params)
        self.sess = tf.Session()
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]

        token_vocab_path = os.path.join(model_dir, constants.FILENAME_TOKEN_VOCAB)
        if os.path.isfile(token_vocab_path):
            logging.info("loading token vocabulary ...")
            self.token_vocab = deeptext.utils.serialization.restore(token_vocab_path)
            logging.info("token vocabulary size = %d", len(self.token_vocab.vocabulary_))
        else:
            self.token_vocab = None

        label_vocab_path = os.path.join(model_dir, constants.FILENAME_LABEL_VOCAB)
        if os.path.isfile(label_vocab_path):
            logging.info("loading label vocabulary ...")
            self.label_vocab = deeptext.utils.serialization.restore(label_vocab_path)
            logging.info("label vocabulary size = %d", len(self.label_vocab.vocabulary_))
        else:
            self.label_vocab = None

    def __del__(self):
        self.sess.close()

    def preprocess(self, training_data_path):

        def tokenizer(iterator):
            for value in iterator:
                yield value

        tokens, labels = read_data(training_data_path)
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]

        if self.token_vocab is None:
            logging.info("generating token vocabulary ...")
            self.token_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
            self.token_vocab.fit(tokens)
            logging.info("token vocabulary size = %d", len(self.token_vocab.vocabulary_))

            token_vocab_path = os.path.join(model_dir, constants.FILENAME_TOKEN_VOCAB)
            deeptext.utils.serialization.save(self.token_vocab, token_vocab_path)

        self.token_ids = self.preprocess_token_transform(tokens)
        self.params[constants.PARAM_KEY_TOKEN_VOCAB_SIZE] = len(self.token_vocab.vocabulary_)

        if self.label_vocab is None:
            logging.info("generating label vocabulary ...")
            self.label_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
            self.label_vocab.fit(labels)
            logging.info("label vocabulary size = %d", len(self.label_vocab.vocabulary_))

            label_vocab_path = os.path.join(model_dir, constants.FILENAME_LABEL_VOCAB)
            deeptext.utils.serialization.save(self.label_vocab, label_vocab_path)

        self.label_ids = self.preprocess_label_transform(labels)
        self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE] = len(self.label_vocab.vocabulary_)

        self.tensor_tokens = tf.placeholder_with_default(self.token_ids, name=constants.TENSOR_NAME_TOKENS, shape=[None,
                                                                                                                   self.params[
                                                                                                                       constants.PARAM_KEY_MAX_DOCUMENT_LEN]])
        self.tensor_labels = tf.placeholder_with_default(self.label_ids, name=constants.TENSOR_NAME_LABELS, shape=[None,
                                                                                                                   self.params[
                                                                                                                       constants.PARAM_KEY_MAX_DOCUMENT_LEN]])

        self.build_model(self.tensor_tokens, self.tensor_labels)

    def preprocess_token_transform(self, tokens):
        token_ids = self.token_vocab.transform(tokens)
        return np.array(list(token_ids))

    def preprocess_label_transform(self, labels):
        label_ids = self.label_vocab.transform(labels)
        return np.array(list(label_ids))

    def build_model(self, x, y):

        TOKEN_VOCAB_SIZE = self.params[constants.PARAM_KEY_TOKEN_VOCAB_SIZE]
        LABEL_VOCAB_SIZE = self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE]
        MAX_DOCUMENT_LEN = self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]
        EMBEDDING_SIZE = self.params[constants.PARAM_KEY_EMBEDDING_SIZE]
        DROPOUT_PROB = self.params[constants.PARAM_KEY_DROPOUT_PROB]

        word_vectors = tf.contrib.layers.embed_sequence(
            x, vocab_size=TOKEN_VOCAB_SIZE, embed_dim=EMBEDDING_SIZE, scope='words')

        cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT_PROB)

        output, _ = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32)
        output = tf.reshape(output, [-1, EMBEDDING_SIZE])
        logits = tf.contrib.layers.fully_connected(output, LABEL_VOCAB_SIZE)
        self.logits_tensor = logits = tf.reshape(logits, [-1, MAX_DOCUMENT_LEN, LABEL_VOCAB_SIZE],
                                                 name=constants.TENSOR_NAME_LOGITS)

        zeros_with_shape = tf.zeros_like(y, dtype=tf.int64)
        weights = tf.to_double(tf.reshape(tf.not_equal(zeros_with_shape, y), [-1]))

        target = tf.one_hot(y, LABEL_VOCAB_SIZE, 1, 0)
        loss = tf.contrib.losses.softmax_cross_entropy(
            tf.reshape(logits, [-1, LABEL_VOCAB_SIZE]),
            tf.reshape(target, [-1, LABEL_VOCAB_SIZE]),
            weights=weights)
        self.tensor_loss = tf.identity(loss, name=constants.TENSOR_NAME_LOSS)
        self.summ_training_loss = tf.summary.scalar("training_loss", self.tensor_loss)
        self.summ_validation_loss = tf.summary.scalar("validation_loss", self.tensor_loss)

        # Create a training op.
        self.tensor_optimizer = tf.contrib.layers.optimize_loss(
            self.tensor_loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=0.01)

        self.tensor_prediction = tf.argmax(logits, 2, name=constants.TENSOR_NAME_PREDICTION)

        self.summ = tf.summary.merge_all()

    def fit(self, steps, batch_size, training_data_path, validation_data_path=None):

        self.preprocess(training_data_path)

        validation_token_ids = None
        validation_label_ids = None
        if validation_data_path is not None:
            tokens, labels = read_data(validation_data_path)
            validation_token_ids = self.preprocess_token_transform(tokens)
            validation_label_ids = self.preprocess_label_transform(labels)

        self.sess.run(tf.global_variables_initializer())
        self.restore(self.sess)

        logdir = constants.SUMMARY_FILE_PATH + '/' + time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime())
        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(self.sess.graph)
        logging.info("logdir: "+logdir)

        for i in xrange(steps):
            curr_row_ids = np.random.choice(self.token_ids.shape[0], batch_size)
            curr_token_ids = self.token_ids[curr_row_ids]
            curr_label_ids = self.label_ids[curr_row_ids]

            self.sess.run(self.tensor_optimizer,
                          feed_dict={self.tensor_tokens: curr_token_ids, self.tensor_labels: curr_label_ids})

            if (i + 1) % 100 == 0:
                c, s = self.sess.run([self.tensor_loss, self.summ_training_loss],
                                     feed_dict={self.tensor_tokens: curr_token_ids, self.tensor_labels: curr_label_ids})
                writer.add_summary(s, i + 1)
                logging.info("step: %d, training loss: %.2f", i + 1, c)

                if validation_data_path is not None:
                    c, s = self.sess.run([self.tensor_loss, self.summ_validation_loss],
                                         feed_dict={self.tensor_tokens: validation_token_ids,
                                                    self.tensor_labels: validation_label_ids})
                    writer.add_summary(s, i + 1)
                    logging.info("step: %d, validation loss: %.2f", i + 1, c)

                self.save(self.sess)

    def predict(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        labels_transform = self.sess.run(self.tensor_prediction, feed_dict={
            self.tensor_tokens: tokens_transform
        })

        labels = []
        for item in self.label_vocab.reverse(labels_transform):
            labels.append(item.split(' '))
        return labels

    def logits(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        logits = self.sess.run(self.logits_tensor, feed_dict={
            self.tensor_tokens: tokens_transform
        })
        entropy_list = []
        for i in xrange(len(tokens)):
            max_entropy = 0
            for j in xrange(len(tokens[i])):
                cur_entropy = entropy(logits[i][j])
                max_entropy = max(cur_entropy, max_entropy)
            entropy_list.append(max_entropy)
        return entropy_list

    def evaluate(self, testing_data_path):
        tokens, labels = read_data(testing_data_path)

        predicted_labels = self.predict(tokens)

        label_corre_cnt = 0
        label_total_cnt = 0
        sentence_corre_cnt = 0
        sentence_total_cnt = len(labels)
        for i in xrange(len(labels)):
            label_total_cnt += len(labels[i])

            sentencen_corre = True
            for j in xrange(min(len(labels[i]), len(predicted_labels[i]))):
                if labels[i][j] == predicted_labels[i][j]:
                    label_corre_cnt += 1
                else:
                    sentencen_corre = False
            if sentencen_corre:
                sentence_corre_cnt += 1

        logging.info("total label count: %d, label accuracy: %.2f", label_total_cnt,
                     1.0 * label_corre_cnt / label_total_cnt)
        logging.info("total sentence count: %d, sentence accuracy: %.2f", sentence_total_cnt,
                     1.0 * sentence_corre_cnt / sentence_total_cnt)
