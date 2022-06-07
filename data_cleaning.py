import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")


def cleaning(string: str) -> str:
    emoticons = re.findall(r'[:|;][-]?[)|(|<>]', string)
    string = re.sub(r'\d', '', string)
    string = re.sub(r'<.*?>', '', string)
    string = re.sub(r'\W(?<!\s)', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = string.lower()
    string = string + str(emoticons)
    return string


def stopword_rem(text: str) -> list:
    return [w for w in text if not w.lower() in stop_words]


def prepare_data(word: str) -> str:
    ps = PorterStemmer()
    return ps.stem(word)


def text_tokenizer(text: str):
    clened = cleaning(text)
    tokens = word_tokenize(clened)
    without_stopwords = stopword_rem(tokens)

    return [prepare_data(w) for w in without_stopwords if len(w) > 3]
