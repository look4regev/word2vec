# word2vec
[![Build Status](https://travis-ci.org/look4regev/word2vec.svg?branch=master)](https://travis-ci.org/look4regev/word2vec)

## Summary
In this project we are using word2vec and the basic 1 hidden layer neural network to train 
a word2vec model. We researched the theory of represeting words as vectors and understood how
to use them in the neural network. We worked with a corpus and thought about the ways to 
clean and process it + making a testset out of it. Finally we used tensorflow to compare.
We ran the algorithm, saw results and documented future improvements.

## Implementing Word2vec with skip-gram for finding similar-context phrases of text.
General example: `./distance TrainedVectors.bin "Chinese river"` will average vectors for words 'Chinese' and 'river' and will return the closest neighbors to the resulting vector:
`Word Cosine distance: Yangtze_River: 0.667376, Yangtze: 0.644091, Qiantang_River: 0.632979`

## Dataset sources:
http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
