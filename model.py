import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion


print()
print("Importing done...")

start_time1 = time.time()
print("Loading files to dataframe...")
# Load files into DataFrames

X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

print("Loading files done...")
start_time2 = time.time()
print("--- %s minutes ---" % np.round(((start_time2 - start_time1)/60),3))
print()

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train['Text'],
        X_train['Score'],
        test_size= 0.2,
        random_state=0
    )

tfidf1 = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1,2))

print("Learning the test set ...")

start_time3 = time.time()

# New Feature transformers

class CharacterCounter(BaseEstimator, TransformerMixin):
    """Count the number of characters in a document."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_characters = X.str.len()
        return n_characters.values.reshape(-1,1) # 2D array
    
class TokenCounter(BaseEstimator, TransformerMixin):
    """Count the number of tokens in a document."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.tokeniser = RegexpTokenizer(r'[A-Za-z]+')
        return self
    
    def transform(self, X):
        n_tokens = X.apply(lambda document: len(self.tokeniser.tokenize(document)))
        return n_tokens.values.reshape(-1,1) # 2D array

#Pipelines

character_pipe = Pipeline([
    ('character_counter', CharacterCounter()),
    ('scaler', MinMaxScaler())
])

token_pipe = Pipeline([
    ('token_counter', TokenCounter()),
    ('scaler', MinMaxScaler())
])

preprocessor = FeatureUnion([
    ('vectoriser', tfidf1),
    ('character', character_pipe),
    ('token', token_pipe)
])

clf = LogisticRegression(solver='newton-cg')
pipe = Pipeline([('Preprocess', preprocessor), ('LR', clf)])

# fitting the model
pipe.fit(X_train, Y_train)
print("Learning done...")
start_time4 = time.time()
print("--- %s minutes ---" % np.round(((start_time4 - start_time3)/60),3))
print()


# Predict the score using the model
print("Predicting...")
start_time5 = time.time()
predicted_by_clr = pipe.predict(X_test)
print("Prediction done...")
start_time6 = time.time()
print("--- %s minutes ---" % np.round(((start_time6 - start_time5)/60),3))


# Evaluate the model on the testing set
print()
print("Accuracy on Combined Feature = ", accuracy_score(Y_test, predicted_by_clr))
print()


#Predict the sample score
print("Predicting submission...")
start_time7 = time.time()
X_submission['Score'] = pipe.predict(X_submission['Text'])
print("Predicting submission done...")
start_time8 = time.time()
print("--- %s minutes ---" % ((start_time8 - start_time7)/60))

# Create the submission file
print("Writing csv file...")
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)

print("Writing csv file done...")
start_time9 = time.time()
print("Total Execution time --- %s minutes ---" % np.round(((start_time9 - start_time1)/60),3))
print()
print()