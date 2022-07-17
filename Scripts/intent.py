import argparse
import warnings
#import numpy as np
#import pandas as pd
#import nltk
import string
import torch

#from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch import cuda
from transformers import BertConfig, AutoTokenizer, TFAutoModelForSequenceClassification
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import scale
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score
#from sklearn.metrics import roc_curve
#from sklearn.linear_model import LogisticRegression
#from sentence_transformers import SentenceTransformer
#from sklearn.ensemble import RandomForestClassifier
#from nltk.corpus import stopwords
#from utils import modelsImport
#models = modelsImport()
warnings.filterwarnings('ignore')

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
#en_stops = set(stopwords.words('english'))


def classifyIntentSbert(model, embed, file):
    """
    :param model: model input for classifier
    :param embed: sentence bert model for sentence embedding
    :param file: test file
    :return: return a csv with predicted output and the rocauc score on test data
    """
    en_stops = set(stopwords.words('english'))
    df = pd.read_csv('Datasets/' + file)
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].apply(lambda i: "".join(t.lower() for t in i if t not in string.punctuation)) # remove punctuations
    df[cols[0]] = df[cols[0]].apply(nltk.tokenize.word_tokenize) # tokenize the sentences
    df[cols[0]] = df[cols[0]].apply(lambda x: [i for i in x if i not in en_stops])  # remove stopwords
    df[cols[0]] = df[cols[0]].apply(lambda x: ' '.join(x))
    df['embeddings'] = df[cols[0]].apply(lambda x: embed.encode(x))
    sent_embeddings = []
    for each in df['embeddings']:
        sent_embeddings.append(each)
    df['pred'] = model.predict(sent_embeddings)
    df.drop('embeddings', inplace=True, axis=1)
    df.to_csv('sbert_output.csv',index=False)
    return roc_auc_score(df[cols[1]], model.predict_proba(sent_embeddings), multi_class='ovr')


def classifyIntentCV(model, embed, file):
    """

    :param model: model input for classifier
    :param embed: CountVectorizer embedding
    :param file: test file
    :return: return a csv with predicted output and the rocauc score on test data
    """
    df = pd.read_csv('Datasets/' + file)
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].apply(lambda i: "".join(t.lower() for t in i if t not in string.punctuation))
    test_vector = embed.transform(df[cols[0]]).toarray()
    df['pred'] = model.predict(test_vector)
    df.to_csv('cv_output.csv',index=False)
    return roc_auc_score(df[cols[1]], model.predict_proba(test_vector), multi_class='ovr')


def classifyIntentSkipgram(model, embed, file):
    """

    :param model: model input for classifier
    :param embed: Word2Vec skipgram embedding
    :param file: test file
    :return: return a csv with predicted output and the rocauc score on test data
    """
    df = df = pd.read_csv('Datasets/' + file)
    cols = df.columns.tolist()
    df[cols[0]] = df[cols[0]].apply(lambda i: "".join(t.lower() for t in i if t not in string.punctuation))
    df[cols[0]] = df[cols[0]].apply(nltk.tokenize.word_tokenize)
    df[cols[0]] = df[cols[0]].apply(lambda x: [i for i in x if i not in en_stops])
    test_vector = np.concatenate([buildWordVector_3i(embed, models.tfidf, z, 100) for z in map(lambda x: x, df[cols[0]])])
    test_vector = scale(test_vector)
    df['pred'] = model.predict(test_vector)
    df.to_csv('output_skipgram.csv',index=False)
    return roc_auc_score(df[cols[1]], model.predict_proba(test_vector), multi_class='ovr')


def sequenceLabeling(model, tokenizer, text):
    """
    Code for sequence labeling BIO tagging on test text
    :param model: pretrained fine tuned BERT model
    :param tokenizer: pretrained fine tuned BERT tokenizer
    :param text: text input
    :return: BIO tags for each token in input
    """
    MAX_LEN = 64
    label_to_id = {'O': 0, 'B-mon': 1, 'I-mon': 2}
    id_to_label = {0: 'O', 1: 'B-mon', 2: 'I-mon'}
    inputs = tokenizer(text,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=MAX_LEN,
                       return_tensors="pt")
    # move to gpu
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    # forward pass
    model = model.to(device)
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size*seq_len,) - predictions at the token level
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id_to_label[i] for i in flattened_predictions.cpu().numpy()]
    prediction = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)
    prediction = [x for x in prediction if x[0] not in ['[CLS]', '[SEP]', '[PAD]']]
    return prediction


def buildWordVector_3i(embed,tfidf, tokens, size):
    """
    # Function to create an averaged document vector when given a list of tokens of the same document
    vec: word2vec model
    tokens: token inputs
    size: size of embedding
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += embed.wv[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


if __name__ == '__main__':
    """
    This part takes arguments for running and checking the output
    For BIO tagging the arguments should contain --model, --text
    For intent classification the arguments should contain --model, --embed, --file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--embed', type=str)
    parser.add_argument('--text', type=str)
    args = parser.parse_args()

    if args.model == 's-labeling':
        tokenizer = DistilBertTokenizerFast.from_pretrained('models/sequenceModel/')
        model = DistilBertForTokenClassification.from_pretrained('models/sequenceModel/')
        print(sequenceLabeling(model, tokenizer, text=args.text))

    if args.embed == 'sbert':
        vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if args.model == 'rf':
            print("ROCAUC Score using sentence BERT and Random Forest:",classifyIntentSbert(models.rf_sbert, vectorizer, args.file))
        if args.model == 'lr':
            print("ROCAUC using sentence BERT and Logistic Regression: ",classifyIntentSbert(models.lr_sbert, vectorizer, args.file))

    if args.embed == 'cv':
        if args.model == 'lr':
            print("ROCAUC Score using Logistic regression and CountVectorizer:",classifyIntentCV(models.lr_cv, models.cv_vectorizer, args.file))
        if args.model == 'rf':
            print("ROCAUC score using Random Forest and CountVectorizer::",classifyIntentCV(models.rf_cv, models.cv_vectorizer, args.file))

    if args.embed == 'sg':
        if args.model == 'lr':
            print("ROCAUC Score using Word2Vec and Logistic Regression:", classifyIntentSkipgram(models.lr_sg, models.sg, args.file))
        if args.model == 'rf':
            print("ROCAUC Score using Word2Vec and Random Forest:", classifyIntentSkipgram(models.rf_sg, models.sg, args.file))

