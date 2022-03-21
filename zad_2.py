import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
cachedStopWords = stopwords.words("english")


def zad_2(string: str) -> str:
    text_tokens = word_tokenize(string)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    print(tokens_without_sw)
