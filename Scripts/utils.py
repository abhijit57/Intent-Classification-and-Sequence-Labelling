"""
This section contains the code for loading and exporting models to intent.py
"""
import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from gensim.models import Word2Vec
#import nltk
#from nltk.corpus import stopwords

train = pd.read_csv('Datasets/three_intents_train_data.csv')
cols = train.columns.tolist()
train[cols[0]] = train[cols[0]].apply(lambda i: "".join(t.lower() for t in i if t not in string.punctuation))
#en_stops = set(stopwords.words('english'))

class modelsImport:
    def __init__(self):
        self.cv_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
        self.cv_vectorizer.fit_transform(train[cols[0]])
        self.sg = self.makeSkipgram()
        self.tfidf = self.maketfidf()

        self.rf_sbert = joblib.load('models/random_forest_sbert_3i.pkl')
        self.lr_sbert = joblib.load('models/logistic_regression_sbert_3i.pkl')

        self.lr_cv = joblib.load('models/logistic_regression_cv_3i.pkl')
        self.rf_cv = joblib.load('models/random_forest_cv_3i.pkl')
        self.rf_sg = joblib.load('models/random_forest_sg_3i.pkl')
        self.lr_sg = joblib.load('models/logistic_regression_sg_3i.pkl')

    def makeSkipgram(self):
        """
        :return: Load and export the word2vec model trained on 3 intents
        """
        sg = Word2Vec.load('models/word2vec_skipgram_3intents.model')
        return sg

    def maketfidf(self):
        """
        :return: tfidf matrix for trained on training corpus
        """
        vectorizer_3i = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
        train[cols[0]] = train[cols[0]].apply(nltk.tokenize.word_tokenize)
        train[cols[0]] = train[cols[0]].apply(lambda x: [i for i in x if i not in en_stops])
        vectorizer_3i.fit_transform([x for x in train[cols[0]]])
        tfidf_3i = dict(zip(vectorizer_3i.get_feature_names_out(), vectorizer_3i.idf_))
        return tfidf_3i




