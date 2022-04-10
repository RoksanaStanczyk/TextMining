import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def cleaning(string: str) -> str:
    emoticons = re.findall(':\)|;\)|;\(|:>|:<|;<|:-\)|;-\)', string)
    string = re.sub(r'\d', '', string)
    string = re.sub('[<>/]', '', string)
    string = re.sub('[,.;:]', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = string.lower()
    string = string + str(emoticons)
    return string


def prepare_data(string: str) -> list:
    stemm_list = []
    porter = PorterStemmer()
    stream = string.split(' ')
    for word in stream:
        stemm_list.append(porter.stem(word))
    return stemm_list


def stopword_rem(text: list) -> list:
    return [word for word in text if not word in stopwords.words()]
