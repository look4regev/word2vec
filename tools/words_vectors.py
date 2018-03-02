import numpy as np


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))


def find_closest(word_query, words_vectors):
    min_dist = 10000  # "Infinity" like init value
    closest_word = None
    for word, vector in words_vectors.iteritems():
        if word_query != word and euclidean_dist(vector, words_vectors[word_query]) < min_dist:
            min_dist = euclidean_dist(vector, words_vectors[word_query])
            closest_word = word
    return closest_word
