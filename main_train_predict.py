import logging
from os.path import join
from glob import glob
from datetime import datetime

import numpy as np

from nn_tools.neural_network import NeuralNetwork
from tools.words_vectors import find_closest
from tools.corpus_manipulator import get_curpos_sentences, get_curpos_words


HIDDEN_LAYER_SIZE = 10
ITERATIONS = 1
L2_REGULARIZATION = 0.1

# TRAINING_SET_FULL_PATHS = glob(join('corpus', 'training-monolingual.tokenized.shuffled', '*.txt'))
# TRAINING_SET_MIN_PATHS = glob(join('corpus', 'training-monolingual.tokenized.shuffled',
#                                    'news.en-00001*'))
TRAINING_SET_SANITY_TEST_PATHS = glob(join('corpus', 'training_sample.txt'))
TRAINING_SET_PATHS = TRAINING_SET_SANITY_TEST_PATHS
WINDOW_SIZE = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_neural_network(curpos_words):
    input_vector_size = output_vector_size = len(curpos_words)
    active_layers_sizes = [HIDDEN_LAYER_SIZE, output_vector_size]
    return NeuralNetwork(input_vector_size, active_layers_sizes, L2_REGULARIZATION)


def get_training_skip_gram(sentences):
    data = []
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            window_start_index = max(word_index - WINDOW_SIZE, 0)
            window_end_index = min(word_index + WINDOW_SIZE, len(sentence)) + 1
            window_sentence = sentence[window_start_index: window_end_index]
            data += [[word, window_word] for window_word in window_sentence if window_word != word]
    return data


def train_pairs_to_one_hot_vectors(train_pairs, word_to_one_hot_vector):
    inputs = [word_to_one_hot_vector[pair[0]] for pair in train_pairs]
    outputs = [word_to_one_hot_vector[pair[1]] for pair in train_pairs]
    return inputs, outputs


def to_one_hot(data_point_index, vector_size):
    temp = np.zeros(vector_size)
    temp[data_point_index] = 1
    return temp


def map_words_to_one_hot_vectors(words):
    word_to_one_hot_vector = {word: to_one_hot(i, len(words)) for i, word in enumerate(words)}
    return word_to_one_hot_vector


def main():
    start_time = datetime.now()

    curpos_words = get_curpos_words(TRAINING_SET_PATHS)
    neural_network = get_neural_network(curpos_words)
    logger.info('Done creating a distinct non-stop-words vocabulary from the corpus')

    sentences = get_curpos_sentences(TRAINING_SET_PATHS)
    train_pairs = get_training_skip_gram(sentences)
    logger.info('Done creating a skip gram training input + labels dataset from the corpus')

    word_to_one_hot_vector = map_words_to_one_hot_vectors(curpos_words)
    logger.info('Done mapping the corpus to one hot vectors')

    inputs, outputs = train_pairs_to_one_hot_vectors(train_pairs, word_to_one_hot_vector)
    logger.info('Done mapping the train set to hot vectors. Starting training.')

    neural_network.train(outputs, inputs, ITERATIONS)
    logger.info('Done training. Starting prediction on test set')

    vocabulary_vectors = {word: neural_network.predict(word_to_one_hot_vector[word],
                                                       return_winning_class=False)
                          for word in curpos_words}
    logger.info('Done prediction. Finding closest words to all vectors')

    for word in vocabulary_vectors:
        closest_word = find_closest(word, vocabulary_vectors)
        print word, '==', closest_word

    end_time = datetime.now()
    run_time_minutes = (end_time - start_time).seconds / 60
    print 'Total run timespan (minutes) =', run_time_minutes


if __name__ == '__main__':
    main()
