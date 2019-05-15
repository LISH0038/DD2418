import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords

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


# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(min_df=3,  max_features=None, \
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',\
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,\
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)

print(tfv.vocabulary_.shape )
print(xtrain_tfv.shape )
print(xvalid_tfv.shape )
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

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