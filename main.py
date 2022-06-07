import pandas as pd
from create_wordclouds import wordclouds
from visualization import describe_data, token_visualization
from classification import training_svm, training_RandomForestClassifier, training_LogisticRegression

df = pd.read_csv('tweets_airline.csv')
df = df[['text', 'airline_sentiment']]

# przedstawienie klas i ich liczby
describe_data(df)
#
wordclouds(df)
token_visualization(df)
training_svm(df)
training_RandomForestClassifier(df)
training_LogisticRegression(df)
