from nltk import word_tokenize
from nltk.stem.porter import *

stemmer = PorterStemmer()


def stemming(string: str) -> list:
    token_words = word_tokenize(string)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
    print(stem_sentence)
