import collections
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from cleaning import cleaning
from cleaning import prepare_data
df_fake = pd.read_csv('News _dataset/Fake.csv')
df_true = pd.read_csv('News _dataset/True.csv')

df_fake = df_fake['text']
df_true = df_true['text']
df_true = cleaning(df_true)
df_fake = cleaning(df_fake)

new_df_true = prepare_data(df_true)
new_df_fake = prepare_data(df_fake)

freq_true = collections.Counter([y for x in new_df_true.values.flatten() for y in x.split()])
freq_fake = collections.Counter([y for x in new_df_fake.values.flatten() for y in x.split()])

wc = WordCloud()
wc.generate_from_frequencies(freq_fake)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
