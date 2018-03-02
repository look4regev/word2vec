from stop_words import get_stop_words


STOP_WORDS = set(get_stop_words('english') +
                 ['the', "'t", ] +
                 ['.', ',', '-', '--', "'s", '(', ')', '!', '|', '"', "'",
                  '?', ':', ';', '~', '`', '^', '&', '_', '+', '/', '*'])


def get_curpos_sentences(dataset_paths):
    sentences = []
    for training_file_path in dataset_paths:
        with file(training_file_path, 'rt') as training_file:
            sentences = [[word for word in sentence.lower().split() if word not in STOP_WORDS]
                         for sentence in training_file.readlines()]
    return sentences


def get_curpos_words(dataset_paths):
    curpos_words = set()
    for training_file_path in dataset_paths:
        with file(training_file_path, 'rt') as training_file:
            file_words = set(training_file.read().lower().split())
            curpos_words.update(file_words)
    return curpos_words.difference(STOP_WORDS)
