import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data_cleaning import text_tokenizer

def describe_data(df):
    category = df.groupby("category")
    category.size().sort_values(ascending=False).plot.bar()
    plt.xlabel("Category")
    plt.ylabel("Number of category")
    plt.show()


def token_visualization(df: pd.DataFrame, category: str):
    df = df['text_original'].loc[df['category'] == category]
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(df)
    col_name = vectorizer.get_feature_names_out()
    array = x_transform.toarray()
    col_sum = np.sum(array, axis=0)
    index = np.argpartition(col_sum, -10)[-10:]

    plt.figure(1)
    plt.barh(col_name[index], col_sum[index])
    plt.title(f'10 najczęściej występujących tokenów w kategorii {category}')
    plt.ylabel('Token')
    plt.xlabel('Ilość')

    vectorizer_tfid = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform_tfid = vectorizer_tfid.fit_transform(df)
    array_tfid = x_transform_tfid.toarray()
    col_sum_tfid = np.sum(array_tfid, axis=0)
    index_tfid = np.argpartition(col_sum_tfid, -10)[-10:]

    plt.figure(2)
    plt.barh(col_name[index_tfid], col_sum_tfid[index_tfid])
    plt.title(f'10 najważniejszych tokenów w kategorii {category}')
    plt.ylabel('Token')
    plt.xlabel('Waga')
    plt.show()
