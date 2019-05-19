import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.svm import SVC
from multiprocessing import cpu_count
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from spacy import attrs
from spacy.symbols import VERB, NOUN, ADV, ADJ
stop_words = stopwords.words('english')

def calculate_accuracy(actual, predicted):
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    rows = actual.shape[0]

    for i in range(0, rows):
        k = np.argmax(predicted[i])
        predicted[i] = np.zeros(3)
        predicted[i][k] = 1

    vsota = np.sum(actual * predicted)
    return 1.0 / rows * vsota 

class WordCountFeatures(TransformerMixin):
    def __init__(self):
        self.wordcount_nb = Pipeline([('text', CountVectorizer()), ('mnb', MultinomialNB())])

    def fit(self, x, y=None):
        self.wordcount_nb.fit(x, y)
        return self
    
    def add_wordcount_features(self, text_series):
        df = pd.DataFrame(text_series.reset_index(drop=True))
        wordcount_features = pd.DataFrame(
            self.wordcount_nb.predict_proba(text_series),
            columns=['nb_pred_' + str(x) for x in self.wordcount_nb.classes_]
                                           )
        del wordcount_features[wordcount_features.columns[0]]
        df = df.merge(wordcount_features, left_index=True, right_index=True)
        return df

    def transform(self, text_series):
        return self.add_wordcount_features(text_series)

class PartOfSpeechFeatures(TransformerMixin):
    def __init__(self):
        self.NLP = spacy.load('en', disable=['parser', 'ner'])
        self.num_cores = cpu_count()

    def part_of_speechiness(self, pos_counts, pos):
        if eval(pos) in pos_counts:
            return pos_counts[eval(pos).numerator]
        return 0

    def add_pos_features(self, df):
        text_series = df["text"]
        df['doc'] = [i for i in self.NLP.pipe(text_series.values, n_threads=self.num_cores)]
        df['pos_counts'] = df['doc'].apply(lambda x: x.count_by(attrs.POS))
        #print(df['pos_counts'])
        df['sentence_length'] = df['doc'].str.len()
        for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            df[pos] = df['pos_counts'].apply(
                lambda x: self.part_of_speechiness(x, pos))
            df[pos] /= df['sentence_length']
        df['avg_word_length'] = (df['doc'].apply(
            lambda x: sum([len(word) for word in x])) / df['sentence_length'])
        del df['pos_counts']
        del df['doc']

        return df

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return self.add_pos_features(df.copy())

class Word2VecFeatures(TransformerMixin):
    def __init__(self):
        self.embeddings_index = {}
        f = open('glove.6B.100d.txt',encoding='utf8')
        for line in tqdm(f):
            values = line.split()
            word = ''.join(values[:-100])
            coefs = np.asarray(values[-100:], dtype='float32')

            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
    def sent2vec(self,s):
        words = str(s).lower()
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(self.embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(100)
        return v / np.sqrt((v ** 2).sum())

    def add_word2vec_features(self, df):
        text_series = df["text"]

        xtrain_glove =[self.sent2vec(x) for x in tqdm(text_series)]
        xtrain_glove = np.array(xtrain_glove)
        word2vec_features = pd.DataFrame(
            xtrain_glove,
            columns=['word_vector' + str(x) for x in range(100)]
                                           )
        df = df.merge(word2vec_features, left_index=True, right_index=True)
        return df

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return self.add_word2vec_features(df.copy())
    
class ClearTextAndNormalize(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        del df['text']
        
        # We choose not to use normalization as we find it does not perform as well
        # as non-normalized data
        
        # Normalize
        # x = df.values 
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x_scaled = min_max_scaler.fit_transform(x)
        # df = pd.DataFrame(x_scaled)

        return df
def run_pipeline(df, pipeline, pipeline_name=''):
    X = pd.Series(df["text"])
    y = preprocessing.LabelEncoder().fit_transform(df.author.values)

    rskf = StratifiedKFold(n_splits=5, random_state=1)
    losses = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict_proba(X_test)
        log_loss = metrics.log_loss(y_test, predictions)
        losses.append(log_loss)
        print(" Log loss: " + str(log_loss))
        print(" Accuracy : %0.3f " % calculate_accuracy(y_test, predictions))

    print(f'{pipeline_name} mean log loss: {round(pd.np.mean(losses), 3)}')

final_solution = Pipeline([
    ('nb', WordCountFeatures()),
    ('pos', PartOfSpeechFeatures()),
    ('w2v', Word2VecFeatures()),
    ('clean', ClearTextAndNormalize()), 
    ('clf', LogisticRegression())
 ])

train_df = pd.read_csv("./train.csv", usecols=["text", "author"])   
run_pipeline(train_df, final_solution)