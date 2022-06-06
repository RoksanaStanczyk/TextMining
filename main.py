import pandas as pd
from data_cleaning import text_tokenizer
from create_wordclouds import wordclouds, wordclouds_category
from visualization import token_visualization, describe_data
from classifiers_training import training_svm, training_RandomForestClassifier, training_LogisticRegression

PATH = 'wordclouds'
# categoty: business, entertainment, politics, sport, tech
CHOOSE_CATEGORY = 'sport'

df = pd.read_csv('Text_Classification.csv')
# print(df)
df = df[['category', 'text_original']]

# wordclouds for all text
wordclouds(df, PATH)

# wordclouds for a given text category
wordclouds_category(df, CHOOSE_CATEGORY, PATH)

describe_data(df)
token_visualization(df, CHOOSE_CATEGORY)
training_svm(df)
training_RandomForestClassifier(df)
training_LogisticRegression(df)