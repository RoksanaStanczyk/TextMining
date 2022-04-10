from nltk.corpus import stopwords
from text_tokenizer import text_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk

df_true = pd.read_csv(r'C:\Users\Roxi0\PycharmProjects\Git\TextMining\News _dataset/True.csv')
df_true = df_true['title']
result = text_tokenizer(str(df_true))
vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(result)
print(X_transform)
