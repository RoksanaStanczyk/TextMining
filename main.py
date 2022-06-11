import pandas as pd
from create_wordclouds import wordclouds, wordclouds_category
from visualization import token_visualization, describe_data
from classifiers_training import training_svm, training_RandomForestClassifier, training_LogisticRegression

# categoty: business, entertainment, politics, sport, tech
CHOOSE_CATEGORY = 'sport'

df = pd.read_csv('Text_Classification.csv')
df = df[['category', 'text_original']]

describe_data(df)

# wordclouds for all text
wordclouds(df)

# wordclouds for a given text category
wordclouds_category(df, CHOOSE_CATEGORY)
token_visualization(df, CHOOSE_CATEGORY)
training_svm(df)
training_RandomForestClassifier(df)
training_LogisticRegression(df)
