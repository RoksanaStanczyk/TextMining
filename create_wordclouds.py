import pandas as pd
import os
from PIL.Image import Image
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from data_cleaning import cleaning
from wordcloud import WordCloud


def wordclouds(df: pd.DataFrame, path: str):
    s_w = set(stopwords.words('english'))
    text_all = " ".join(text for text in df.text_original.astype(str))
    text_all = cleaning(text_all)
    wc = WordCloud()
    wc.generate(text_all)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    wc.to_file(f'{path}/wc_all.png')
    plt.figure('wordclouds for the whole text')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def wordclouds_category(df: pd.DataFrame, category: str, path: str):
    s_w = set(stopwords.words('english'))
    df = df.loc[df['category'] == category]
    text_all = " ".join(text for text in df.text_original.astype(str))
    text_all = cleaning(text_all)
    wc = WordCloud()
    wc.generate(text_all)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    wc.to_file(f'{path}/wc_{category}.png')
    plt.figure(f"wordclouds for {category} category")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
