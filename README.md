# word2vec
[![Build Status](https://travis-ci.org/look4regev/word2vec.svg?branch=master)](https://travis-ci.org/look4regev/word2vec)
## Implementing Word2vec with skip-gram for finding similar-context phrases of text.
General example: `./distance TrainedVectors.bin "Chinese river"` will average vectors for words 'Chinese' and 'river' and will return the closest neighbors to the resulting vector:
`Word Cosine distance: Yangtze_River: 0.667376, Yangtze: 0.644091, Qiantang_River: 0.632979`

## Dataset sources:
http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
