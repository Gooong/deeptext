import os
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn
from gensim.models import Word2Vec

from scipy.stats import entropy

from deeptext.models.base import Base
import deeptext.utils.serialization
import deeptext.models.constants as constants

from utils import read_data


class Word2VecBiCrfSequenceLabeling(Base):
    def __init__(self, params):
        super(Word2VecBiCrfSequenceLabeling, self).__init__(params)
        self.sess = tf.Session()
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]

        label_vocab_path = os.path.join(model_dir, constants.FILENAME_LABEL_VOCAB)
        if os.path.isfile(label_vocab_path):
            logging.info("loading label vocabulary ...")
            self.label_vocab = deeptext.utils.serialization.restore(label_vocab_path)
            logging.info("label vocabulary size = %d", len(self.label_vocab.vocabulary_))
        else:
            self.label_vocab = None

        word2vec_model_filename = os.path.join(model_dir, constants.FILENAME_WORD2VEC_MODEL)
        self.word2vec_model = Word2Vec.load(word2vec_model_filename)
        self.params[constants.PARAM_KEY_EMBEDDING_SIZE] = self.word2vec_model.vector_size

    def __del__(self):
        self.sess.close()

    def preprocess(self, training_data_path):

        def tokenizer(iterator):
            for value in iterator:
                yield value

        tokens, labels = read_data(training_data_path)
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]

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

        self.tokens = tokens
        print constants.TENSOR_NAME_TOKENS
        print self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]
        print self.params[constants.PARAM_KEY_EMBEDDING_SIZE]
        self.tensor_tokens = tf.placeholder(dtype=tf.float32, name=constants.TENSOR_NAME_TOKENS,
                                                         shape=[None,
                                                                self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN],
                                                                self.params[constants.PARAM_KEY_EMBEDDING_SIZE]])
        self.tensor_labels = tf.placeholder_with_default(self.label_ids, name=constants.TENSOR_NAME_LABELS,
                                                         shape=[None,
                                                                self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]])

        print 'build_model'
        self.build_model(self.tensor_tokens, self.tensor_labels)

    def get_token_vectors(self, tokens):
        MAX_DOCUMENT_LEN = self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]
        vectors = []
        for token_list in tokens:
            vector_list = []
            for i in range(MAX_DOCUMENT_LEN):
                if i < len(token_list) and self.word2vec_model.wv.vocab.get(token_list[i]):
                    vector_list.append(list(self.word2vec_model.wv[token_list[i]]))
                else:
                    # print token
                    vector_list.append([0]*self.word2vec_model.vector_size)
            vectors.append(vector_list)

        return np.array(vectors)

    def preprocess_label_transform(self, labels):
        label_ids = self.label_vocab.transform(labels)
        return np.array(list(label_ids))

    def build_model(self, x, y):

        LABEL_VOCAB_SIZE = self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE]
        MAX_DOCUMENT_LEN = self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]
        EMBEDDING_SIZE = self.params[constants.PARAM_KEY_EMBEDDING_SIZE]
        DROPOUT_PROB = self.params[constants.PARAM_KEY_DROPOUT_PROB]

        word_vectors = x
        fw_cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1-DROPOUT_PROB)
        bw_cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=1-DROPOUT_PROB)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_vectors, dtype=tf.float32)
        output = tf.concat(outputs, 2)
        output = tf.reshape(output, [-1, 2 * EMBEDDING_SIZE])

        logits = tf.contrib.layers.fully_connected(output, LABEL_VOCAB_SIZE)
        self.logits_tensor = logits = tf.reshape(logits, [-1, MAX_DOCUMENT_LEN, LABEL_VOCAB_SIZE],
                                                 name=constants.TENSOR_NAME_LOGITS)
        self.sequence_lengths = tf.count_nonzero(y, axis=1)
        log_likelihood, self.tensor_transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf.cast(y, tf.int32),
                                                                                          self.sequence_lengths)

        loss = tf.reduce_mean(-log_likelihood)
        self.tensor_loss = tf.identity(loss, name=constants.TENSOR_NAME_LOSS)
        self.summ_training_loss = tf.summary.scalar("training_loss", self.tensor_loss)
        self.summ_validation_loss = tf.summary.scalar("validation_loss", self.tensor_loss)

        # Create a training op.
        self.tensor_optimizer = tf.contrib.layers.optimize_loss(
            self.tensor_loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=0.01)

        self.summ = tf.summary.merge_all()

    def fit(self, steps, batch_size, training_data_path, validation_data_path=None):

        self.preprocess(training_data_path)
        print 'fit'
        validation_token_vectors = None
        validation_label_ids = None
        if validation_data_path is not None:
            tokens, labels = read_data(validation_data_path)
            validation_token_vectors = self.get_token_vectors(tokens)
            validation_label_ids = self.preprocess_label_transform(labels)

        self.sess.run(tf.global_variables_initializer())
        self.restore(self.sess)

        logdir = constants.SUMMARY_FILE_PATH + '/' + time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime())
        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(self.sess.graph)
        logging.info("logdir: " + logdir)

        for i in xrange(steps):
            curr_row_ids = np.random.choice(range(len(self.tokens)), batch_size)
            curr_tokens = []
            for id in curr_row_ids:
                curr_tokens.append(self.tokens[id])
            curr_token_vectors = self.get_token_vectors(curr_tokens)
            curr_label_ids = self.label_ids[curr_row_ids]
            self.transition_matrix, _ = self.sess.run([self.tensor_transition_params, self.tensor_optimizer],
                                                      feed_dict={self.tensor_tokens: curr_token_vectors,
                                                                 self.tensor_labels: curr_label_ids})

            if (i + 1) % 100 == 0:
                c, s = self.sess.run([self.tensor_loss, self.summ_training_loss],
                                     feed_dict={self.tensor_tokens: curr_token_vectors, self.tensor_labels: curr_label_ids})
                writer.add_summary(s, i + 1)
                logging.info("step: %d, training loss: %.2f", i + 1, c)

                if validation_data_path is not None:
                    c, s = self.sess.run([self.tensor_loss, self.summ_validation_loss],
                                         feed_dict={self.tensor_tokens: validation_token_vectors,
                                                    self.tensor_labels: validation_label_ids})
                    writer.add_summary(s, i + 1)
                    logging.info("step: %d, validation loss: %.2f", i + 1, c)

                self.save(self.sess)

    def predict(self, tokens):
        tokens_vectors = self.get_token_vectors(tokens)
        sentence_lengths = []
        for token_list in tokens:
            sentence_lengths.append(len(token_list))

        tf_unary_scores = self.sess.run(
            self.logits_tensor,
            feed_dict={self.tensor_tokens: tokens_vectors}
        )

        labels = []
        for tf_unary_score_, sentence_length_ in zip(tf_unary_scores, sentence_lengths):
            tf_unary_score_ = tf_unary_score_[:sentence_length_]
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_score_, self.transition_matrix)
            labels.append(list(self.label_vocab.reverse([viterbi_sequence])))
        return labels

    def logits(self, tokens):
        tokens_vectors = self.get_token_vectors(tokens)
        logits = self.sess.run(self.logits_tensor, feed_dict={
            self.tensor_tokens: tokens_vectors
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

        tokens_vectors = self.get_token_vectors(tokens)
        labels_transform = self.preprocess_label_transform(labels)

        tf_unary_scores, tf_transition_params, tf_sequence_lengths = self.sess.run(
            [self.logits_tensor, self.tensor_transition_params, self.sequence_lengths],
            feed_dict={
                self.tensor_tokens: tokens_vectors
            })

        label_corre_cnt = 0
        label_total_cnt = 0
        sentence_corre_cnt = 0
        sentence_total_cnt = len(labels)

        for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, labels_transform, tf_sequence_lengths):
            # Remove padding from the scores and tag sequence.
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params)

            # Evaluate accuracy.
            sentence_corre = True
            label_total_cnt += sequence_length_
            sencence_label_corre_ = np.sum(np.equal(viterbi_sequence, y_))
            label_corre_cnt += sencence_label_corre_

            if sencence_label_corre_ == sequence_length_:
                sentence_corre_cnt += 1

        logging.info("total label count: %d, label accuracy: %.2f", label_total_cnt,
                     1.0 * label_corre_cnt / label_total_cnt)
        logging.info("total sentence count: %d, sentence accuracy: %.2f", sentence_total_cnt,
                     1.0 * sentence_corre_cnt / sentence_total_cnt)
