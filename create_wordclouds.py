import pandas as pd
from matplotlib import pyplot as plt
from data_cleaning import text_tokenizer
from wordcloud import WordCloud


def wordclouds(df: pd.DataFrame):
    text_all = " ".join(text for text in df.text.astype(str))
    text_all = text_tokenizer(text_all)
    text_all = " ".join(i for i in text_all)
    wc = WordCloud()
    wc.generate(text_all)
    plt.figure('wordclouds for the whole text')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def wordclouds_category(df: pd.DataFrame, category: str):
    df = df.loc[df['airline_sentiment'] == category]
    text_all = " ".join(text for text in df.text.astype(str))
    text_all = text_tokenizer(text_all)
    text_all = " ".join(i for i in text_all)
    wc = WordCloud()
    wc.generate(text_all)
    plt.figure(f"wordclouds for {category} category")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
