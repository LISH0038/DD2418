import pandas as pd
import numpy as np
import xgboost as xgb
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
TEXT_COLUMN = 'text'
Y_COLUMN = 'author'
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def multiclass_accurate(actual, predicted):
    """Multi class version of accurate.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one hot result per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    rows = actual.shape[0]
    vsota = np.sum(actual * predicted)
    return 1.0 / rows * vsota


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#sample = pd.read_csv('../input/sample_submission.csv')

#print(train.author.head())
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)


xtrain, xvalid, ytrain, yvalid = model_selection.train_test_split(train.text.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)


class UnigramPredictions(TransformerMixin):
    def __init__(self):
        self.unigram_mnb = Pipeline([('text', CountVectorizer()), ('mnb', MultinomialNB())])

    def fit(self, x, y=None):
        # Every custom transformer requires a fit method. In this case, we want to train
        # the naive bayes model.
        self.unigram_mnb.fit(x, y)
        return self
    
    def add_unigram_predictions(self, text_series):
        # Resetting the index ensures the indexes equal the row numbers.
        # This guarantees nothing will be misaligned when we merge the dataframes further down.
        df = pd.DataFrame(text_series.reset_index(drop=True))
        # Make unigram predicted probabilities and label them with the prediction class, aka 
        # the author.
        unigram_predictions = pd.DataFrame(
            self.unigram_mnb.predict_proba(text_series),
            columns=['naive_bayes_pred_' + str(x) for x in self.unigram_mnb.classes_]
                                           )
        # We only need 2 out of 3 columns, as the last is always one minus the 
        # sum of the other two. In some cases, that colinearity can actually be problematic.
        del unigram_predictions[unigram_predictions.columns[0]]
        df = df.merge(unigram_predictions, left_index=True, right_index=True)
        return df

    def transform(self, text_series):
        # Every custom transformer also requires a transform method. This time we just want to 
        # provide the unigram predictions.
        return self.add_unigram_predictions(text_series)

class TfidfFeatures():
    def __init__(self):
        self.tfv = TfidfVectorizer(min_df=3,  max_features=None, \
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',\
                    ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,\
                    stop_words = 'english')
        self.tfidfpipeline = Pipeline([('tfv', tfv), ('mnb', MultinomialNB())])

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        # self.tfv.fit(list(xtrain) + list(xvalid))
        # xtrain_tfv =  self.tfv.transform(xtrain) 
        # xvalid_tfv = self.tfv.transform(xvalid)

        # # Fitting a simple Logistic Regression on TFIDF
        # clf = LogisticRegression(C=1.0)
        # clf.fit(xtrain_tfv, ytrain)
        # predictions = clf.predict_proba(xvalid_tfv)

    def fit(self, x, y=None):
        # Every custom transformer requires a fit method. In this case, we want to train
        # the naive bayes model.
        self.tfv.fit(x, y)
        return self
    
    def add_unigram_predictions(self, text_series):
        # Resetting the index ensures the indexes equal the row numbers.
        # This guarantees nothing will be misaligned when we merge the dataframes further down.
        df = pd.DataFrame(text_series.reset_index(drop=True))
        # Make unigram predicted probabilities and label them with the prediction class, aka 
        # the author.
        tf_idf = pd.DataFrame(
            self.tfv.predict_proba(text_series),
            columns=['tf_idf' + str(x) for x in self.tfv.classes_]
                                           )
        # We only need 2 out of 3 columns, as the last is always one minus the 
        # sum of the other two. In some cases, that colinearity can actually be problematic.
        del tf_idf[tf_idf.columns[0]]
        df = df.merge(tf_idf, left_index=True, right_index=True)
        return df

    def transform(self, text_series):
        # Every custom transformer also requires a transform method. This time we just want to 
        # provide the unigram predictions.
        return self.add_unigram_predictions(text_series)

NLP = spacy.load('en', disable=['parser', 'ner'])
class PartOfSpeechFeatures(TransformerMixin):
    def __init__(self):
        self.NLP = NLP
        # Store the number of cpus available for when we do multithreading later on
        self.num_cores = cpu_count()

    def part_of_speechiness(self, pos_counts, part_of_speech):
        if eval(part_of_speech) in pos_counts:
            return pos_counts[eval(part_of_speech).numerator]
        return 0

    def add_pos_features(self, df):
        text_series = df[TEXT_COLUMN]
        """
        Parse each sentence with part of speech tags. 
        Using spaCy's pipe method gives us multi-threading 'for free'. 
        This is important as this is by far the single slowest step in the pipeline.
        If you want to test this for yourself, you can use:
            from time import time 
            start_time = time()
            (some code)
            print(f'Code took {time() - start_time} seconds')
        For faster functions the timeit module would be standard... but that's
        meant for situations where you can wait for the function to be called 1,000 times.
        """
        df['doc'] = [i for i in self.NLP.pipe(text_series.values, n_threads=self.num_cores)]
        df['pos_counts'] = df['doc'].apply(lambda x: x.count_by(attrs.POS))
        # We get a very minor speed boost here by using pandas built in string methods
        # instead of df['doc'].apply(len). String processing is generally slow in python,
        # use the pandas string methods directly where possible.
        df['sentence_length'] = df['doc'].str.len()
        # This next step generates the fraction of each sentence that is composed of a 
        # specific part of speech.
        # There's admittedly some voodoo in this step. Math can be more highly optimized in python
        # than string processing, so spaCy really stores the parts of speech as numbers. If you
        # try >>> VERB in the console you'll get 98 as the result.
        # The monkey business with eval() here allows us to generate several named columns
        # without specifying in advance that {'VERB': 98}.
        for part_of_speech in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            df[f'{part_of_speech.lower()}iness'] = df['pos_counts'].apply(
                lambda x: self.part_of_speechiness(x, part_of_speech))
            df[f'{part_of_speech.lower()}iness'] /= df['sentence_length']
        df['avg_word_length'] = (df['doc'].apply(
            lambda x: sum([len(word) for word in x])) / df['sentence_length'])
        return df

    def fit(self, x, y=None):
        # since this transformer doesn't train a model, we don't actually need to do anything here.
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

        # Store the number of cpus available for when we do multithreading later on
    def add_word2vec_features(self, df):
        text_series = df[TEXT_COLUMN]

        #df = pd.DataFrame(text_series.reset_index(drop=True))
        xtrain_glove =[self.sent2vec(x) for x in tqdm(text_series)]
        xtrain_glove = np.array(xtrain_glove)
        word2vec_features = pd.DataFrame(
            xtrain_glove,
            columns=['tf_idf' + str(x) for x in range(100)]
                                           )
        # We only need 2 out of 3 columns, as the last is always one minus the 
        # sum of the other two. In some cases, that colinearity can actually be problematic.
        del word2vec_features[word2vec_features.columns[0]]
        #print(word2vec_features)
        #print(np.shape(word2vec_features))
        df = df.merge(word2vec_features, left_index=True, right_index=True)
        return df

    def fit(self, x, y=None):
        # since this transformer doesn't train a model, we don't actually need to do anything here.
        return self

    def transform(self, df):
        return self.add_word2vec_features(df.copy())
    # Always start with these features. They work (almost) everytime!
    def cache(self):
        
        #print(predictions)
        print(predictions.shape)
        print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

        ptest = np.zeros((1958, 3))
        pp = np.random.randint(3, size = 1958)
        for i in range(0, 1958):
            ptest[i][pp[i]] = 1
        print ("logloss: %0.3f " % multiclass_logloss(yvalid, ptest))
        print("accurate: %0.3f " % multiclass_accurate(yvalid, ptest))

        for i in range(0, 1958):
            k = np.argmax(predictions[i])
            predictions[i] = np.zeros(3)
            predictions[i][k] = 1
        print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
        print("accurate: %0.3f " % multiclass_accurate(yvalid, predictions))

        #Naive Bayes
        ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), stop_words = 'english')

        # Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
        ctv.fit(list(xtrain) + list(xvalid))
        xtrain_ctv =  ctv.transform(xtrain) 
        xvalid_ctv = ctv.transform(xvalid)
        clf = MultinomialNB()
        clf.fit(xtrain_ctv, ytrain)
        predictions = clf.predict_proba(xvalid_ctv)
        print(predictions)
        print ("NB：　logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
        print("NB: %0.3f " % multiclass_accurate(yvalid, predictions))

        
        # this function creates a normalized vector for the whole sentence
        
        xtrain_glove =[self.sent2vec(x) for x in tqdm(xtrain)]
        xvalid_glove = [self.sent2vec(x) for x in tqdm(xvalid)]
        xtrain_glove = np.array(xtrain_glove)
        xvalid_glove = np.array(xvalid_glove)
        # Fitting a simple xgboost on glove features
        #clf = xgb.XGBClassifier(nthread=10, silent=False)
        clf = LogisticRegression(C=1.0)
        clf.fit(xtrain_glove, ytrain)
        predictions = clf.predict_proba(xvalid_glove)
        print(predictions)
        print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
        print("word2vec : %0.3f " % multiclass_accurate(yvalid, predictions))

class DropStringColumns(TransformerMixin):
    # You may have noticed something odd about this class: there's no __init__!
    # It's actually inherited from TransformerMixin, so it doesn't need to be declared again.
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype == object:
                del df[col]
        return df
def test_pipeline(df, nlp_pipeline, pipeline_name=''):
    #y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(df.author.values)

    # If you've done EDA, you may have noticed that the author classes aren't quite balanced.
    # We'll use stratified splits just to be on the safe side.
    rskf = StratifiedKFold(n_splits=5, random_state=1)
    losses = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        predictions = nlp_pipeline.predict_proba(X_test)
        losses.append(metrics.log_loss(y_test, predictions))
        print(" Accuracy : %0.3f " % multiclass_accurate(y_test, predictions))

    print(f'{pipeline_name} kfolds log losses: {str([str(round(x, 3)) for x in sorted(losses)])}')
    print(f'{pipeline_name} mean log loss: {round(pd.np.mean(losses), 3)}')

logit_all_features_pipe = Pipeline([
    
        ('uni', UnigramPredictions()),
        ('nlp', PartOfSpeechFeatures()),
        ('word2vec', Word2VecFeatures()),
     ('clean', DropStringColumns()), 
     ('clf', LogisticRegression())
 ])
# logit_all_features_pipe.fit(xtrain, ytrain)
# predictions = logit_all_features_pipe.predict_proba(xvalid)
# print(predictions)
# print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
                                   

def generate_submission_df(trained_prediction_pipeline, test_df):
    predictions = pd.DataFrame(
        trained_prediction_pipeline.predict_proba(test_df.text),
        columns=trained_prediction_pipeline.classes_
                               )
    predictions['id'] = test_df['id']
    predictions.to_csv("submission.csv", index=False)
    return predictions
train_df = pd.read_csv("./train.csv", usecols=[TEXT_COLUMN, Y_COLUMN])   
test_pipeline(train_df, logit_all_features_pipe)