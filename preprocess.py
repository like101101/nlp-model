import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#Author: Ke Li
#This is the template for preprocessing data frame

def process(df):
    wnl = WordNetLemmatizer()
    df['Text'] = df['Text'].fillna(' ')
    df['Summary'] = df['Summary'].fillna(' ')
    df['Text'] =  df['Summary'] + " " + df['Text']
    print('Applying Tokenize and Stem')
    df['Text'] = df['Text'].apply(word_tokenize).apply(lambda x: [wnl.lemmatize(y) for y in x]).apply(" ".join)
    return df


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)
