from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")


def cleaning(df):
    df = df.str.lower()
    df = df.str.replace(r'[0-9]', ' ', regex=True)
    df = df.str.replace(r'<.*?>', ' ', regex=True)
    df = df.str.replace(r'[\W]', ' ', regex=True)
    df = df.str.replace(r' +', ' ', regex=True)
    return df



def prepare_data(df):
    df = df.str.split()
    df['stemmed'] = df.apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.
    df['stemmed'] = df.stemmed.apply(', '.join)
    return df['stemmed']