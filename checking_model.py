

# Data preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle
import string
import re
import nltk
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import word_tokenize

Data = pd.read_csv('news_data.csv')
data = Data.iloc[:, 2:4]

#Transform label to binary
#data.label = data.label.map(dict(REAL=1, FAKE=0))


# Visualising the dataset
import seaborn as sb
def create_distribution(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='hls')     
#create_distribution(data)
data =  shuffle(data)

# Clean the text
eng_stemmer2 = PorterStemmer()
eng_stemmer = SnowballStemmer('english')
#nltk.corpus.stopwords.words('english').remove('not')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.remove('not')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed
#tokens = data.iloc[3, 0]
def process_data(tokens):
    tokens = re.sub(r"what’s", "what is ", tokens)
    tokens = re.sub(r"\’s", " ", tokens)
    tokens = re.sub(r"\'ve", " have ", tokens)
    tokens = re.sub(r"can’t", "cannot ", tokens)
    tokens = re.sub(r"don’t", "do not ", tokens)
    tokens = re.sub(r"doesn’t", "do not ", tokens)
    tokens = re.sub(r"n’t", " not ", tokens)
    tokens = re.sub(r"i’m", "i am ", tokens)
    tokens = re.sub(r"\'re", " are ", tokens)
    tokens = re.sub(r"\’d", " would ", tokens)
    tokens = re.sub(r"\’ll", " will ", tokens)
    tokens = re.sub(r"what's", "what is ", tokens)
    tokens = re.sub(r"\'s", " ", tokens)
    tokens = re.sub(r"\'ve", " have ", tokens)
    tokens = re.sub(r"can't", "cannot ", tokens)
    tokens = re.sub(r"don't", "do not ", tokens)
    tokens = re.sub(r"doesn't", "do not ", tokens)
    tokens = re.sub(r"n't", " not ", tokens)
    tokens = re.sub(r"i'm", "i am ", tokens)
    tokens = re.sub(r"\'re", " are ", tokens)
    tokens = re.sub(r"\'d", " would ", tokens)
    tokens = re.sub(r"\'ll", " will ", tokens)
    
    tokens = re.sub(r'[\n\t]', ' ', tokens)
    tokens = re.sub(r"\0s", "0", tokens)
    tokens = re.sub(r"e - mail", "email", tokens)
    
    tokens = re.sub('[' + string.punctuation  + ']', ' ', tokens)
    tokens = re.sub(r'[0-9]+', ' ', tokens)
    
    tokens = re.sub("—", " ", tokens)
    tokens = re.sub("’", " ", tokens)
    tokens = re.sub("„", " ", tokens)
    tokens = re.sub("“", " ", tokens)
    tokens = re.sub("”", " ", tokens)
    tokens = tokens.lower()
    tokens = tokens.split()
    tokens = [w for w in tokens if w not in stopwords and len(w) >= 3 ]

    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)

    return tokens_stemmed

# Creating corpus1
corpus = []
for i,j in data.iterrows():
    text = data['text'][i]
    text = process_data(text)
    text = ' '.join(text)
    corpus.append(text)

# Sentiment feature
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def sen_feature(data):
    vs = analyzer.polarity_scores(data)
    return vs['compound']
sen_mat = []
for i in range(len(corpus)):
    compound_score = sen_feature(corpus[i])
    sen_mat.append(compound_score)
sen_mat_ndarray = np.asarray(sen_mat)
sen_mat_ndarray = sen_mat_ndarray.transpose()
sen_mat_ndarray = sen_mat_ndarray.reshape((6335, 1))

# CV, Tf-idf feature 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv = CountVectorizer()
X = cv.fit_transform(corpus)
y = data.iloc[:, 1]

tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range  = (1,3))
X = tfidf_ngram.fit_transform(corpus)
y = data.iloc[:, 1]

# Combining Sentiment Feature to matrix feature X
import scipy as sp
X = sp.sparse.hstack((X, sen_mat_ndarray),format='csr')


# Spliting corpus1 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Embedding matrix feature
new_corpus = []
for text in corpus:
    new_corpus.append(text.split())
    
    """
# GloVe - 2 long to wait .. need to improve
import csv
words = pd.read_table('glove.6B.50d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
def vec(w):
  return np.asarray(words.loc[w])

def vec_sen(sentence):
    mean = np.zeros((50,1))
    n=0
    for i in range(len(sentence)):
        try:
            vec(sentence[i])
        except KeyError:
           x = np.zeros((50,1))
        else:
            x = vec(sentence[i])
            x = x.reshape((50, 1))
            n += 1
        mean += x
    return mean / n
w2v_mat = []
for sentence in new_corpus:
    w2v_mat.append(vec_sen(sentence))
# Storing the w2v
import pickle
f = open('store.pckl', 'wb')
pickle.dump(w2v_mat, f)
f.close()
f = open('store.pckl', 'rb')
w2v_mat = pickle.load(f)
f.close()"""

# RNN for corpus2
data2 = Data.iloc[:, [1, 3] ]
data2.Label = data2.Label.map(dict(REAL=1, FAKE=0))

corpus2 = []
for i,j in data2.iterrows():
    text = data2['title'][i]
    text = process_data(text)
    text = ' '.join(text)
    corpus2.append(text)

# Split the corpus2's training data 
from sklearn.cross_validation import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(corpus2, data2.Label, test_size=0.2, random_state = 0)

# Creating the embedding matrix
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
import tokenize

MAX_NB_WORDS = 12000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train2+X_test2)
train_sequence = tokenizer.texts_to_sequences(X_train2)
test_sequence = tokenizer.texts_to_sequences(X_test2)
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
#print('Found %s word vectors of word2vec' % len(word2vec.vocab))
word_index = tokenizer.word_index
#print('Found %s unique tokens' % len(word_index))
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D,Input,Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
embedding_layer = Embedding(nb_words,
                            300,
                            input_length = 50,
                            weights=[embedding_matrix]
                            )
model = Sequential()
model.add(embedding_layer )
model.add(LSTM(150))
model.add(Dropout(0.25))
model.add(Dense(units = 1, activation='sigmoid'))
#model.add(Dense(units = 1, activation='relu')), sigmoid is better
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

train_sequence = sequence.pad_sequences(train_sequence, maxlen=50)
test_sequence = sequence.pad_sequences(test_sequence, maxlen=50)
history = model.fit(train_sequence, y_train2, validation_data=(test_sequence, y_test2), epochs= 30 , batch_size = 128)

# Evaluating the RNN model
bst_val_score = sum(history.history['val_acc'])/20
preds = model.predict([test_sequence], batch_size=1267, verbose=1)
print('Mean of validation accuracies :' + str(bst_val_score))
print('Preds test set                :' + str(preds))

# Plot confusion matrix funct
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

# Get False Postitive Rate 
def get_fpr(cnf_matrix):
    FP = cm[1, 0]
    FN = cm[0, 1]
    TP = cm[0, 0]
    TN = cm[1, 1]
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    False_Alarm_Rate = FP/(TN + FP)
    return False_Alarm_Rate

# Testing the model with distinc features // dont need to run all of this
# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
y_pred = clf_nb.predict(X_test)
y_pred_2 = clf_nb.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_nb = metrics.accuracy_score(y_test, y_pred)
fpr_nb = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

#Logistic Regression
from sklearn.linear_model import  LogisticRegression
clf_lr = LogisticRegression(penalty="l2",C=1)
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_lr = metrics.accuracy_score(y_test, y_pred)
fpr_lr = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

#SGD
from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)
clf_sgd.fit(X_train, y_train)
y_pred = clf_sgd.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_sgd = metrics.accuracy_score(y_test, y_pred)
fpr_sgd = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

#SVM
from sklearn.svm import SVC
clf_svm = SVC(kernel = 'linear')
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_svm = metrics.accuracy_score(y_test, y_pred)
fpr_svm = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

#Passive aggressive
from sklearn.linear_model import PassiveAggressiveClassifier

clf_pac = PassiveAggressiveClassifier(C = 1.1 ,n_iter=125)
clf_pac.fit(X_train, y_train)

y_pred = clf_pac.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracy_pac = metrics.accuracy_score(y_test, y_pred)
fpr_pac = get_fpr(cm)

plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

#K-NN
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
clf_knn.fit(X_train, y_train)
y_pred = clf_knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_knn = metrics.accuracy_score(y_test, y_pred)
fpr_knn = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

# Fit Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf_tree.fit(X_train, y_train)
y_pred = clf_tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_tree = metrics.accuracy_score(y_test, y_pred)
fpr_tree = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

# XGBoost
import xgboost
from xgboost.sklearn import XGBClassifier
clf_xgb = XGBClassifier()
clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_xgb = metrics.accuracy_score(y_test, y_pred)
fpr_xgb = get_fpr(cm)
plot_confusion_matrix(cm, classes=['REAL', 'FAKE'])

# Applying K-fold to get accuracy
from sklearn.model_selection import cross_val_score
def kfold_accuracy(clf):
    accuracies = cross_val_score(estimator = clf, X = X, y = y,  cv = 6)
    return accuracies.mean()
kfold_accuracy_nb = kfold_accuracy(clf_nb)
kfold_accuracy_lr = kfold_accuracy(clf_lr)
kfold_accuracy_sgd = kfold_accuracy(clf_sgd)
kfold_accuracy_svm = kfold_accuracy(clf_svm)
kfold_accuracy_pac = kfold_accuracy(clf_pac)
kfold_accuracy_knn = kfold_accuracy(clf_knn)
kfold_accuracy_tree = kfold_accuracy(clf_tree)
kfold_accuracy_xgb = kfold_accuracy(clf_xgb)

# Grid search to find PAC's best parameters
parameters = {'C': [0.8, 0.9, 0.95, 1.0, 1.05,  1.1],
              'n_iter' : [75, 100, 125, 150, 200]
}

from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(clf_pac, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
# """     result C = 1.1 , n_inter = 125      """
    
# Saved the PAC model to .sav file
import pickle
model_file = 'pac_model.sav'
pickle.dump(clf_pac,open(model_file,'wb'))

# Saved the tfidf to transform input
tfidf_file = 'tfidf.sav'
pickle.dump(tfidf_ngram,open(tfidf_file, 'wb'))

# Fun thing
sen_test = "In 2013, Clinton told Goldman Sachs bigwigs: \
        'I would like to see people like Donald Trump run for office.\
        They're honest, and can't be bought"

sentiment_score = sen_feature(sen_test)
X_sen = tfidf_ngram.transform([sen_test])
X_sen = sp.sparse.hstack((X_sen, np.array([sentiment_score])),format='csr')
label_sen = clf_pac.predict(X_sen)
proba_truth_sen = clf_pac.decision_function(X_sen)[0]
proba_doubt_sen = (1 - abs(proba_truth_sen))*abs(proba_truth_sen)/(-proba_truth_sen)
print(" PAC_model :")
print(" This new is : " + label_sen[0])
print(" The sentiment score         :" + str(sentiment_score))
print(" The truth score             :" + str(proba_truth_sen))
print(" The doublt score            :" + str(proba_doubt_sen))

X_sen2 = tokenizer.texts_to_sequences([sen_test])
X_sen2 = sequence.pad_sequences(X_sen2, maxlen=50)

label_sen2 = model.predict(X_sen2)
if label_sen2[0] < 0.5 :
    label_sen2 = 'FAKE'
else :
    label_sen2 = 'REAL'
proba_truth_sen2 = model.predict_proba(X_sen2)
print(" RNN_model : ")
print(" This new is : " + label_sen2)
print(" The sentiment score         :" + str(sentiment_score))
print(" The truth score             :" + str(proba_truth_sen2[0][0]))
print(" The doublt score            :" + str(1- proba_truth_sen2[0][0]))

if __name__ == '__main__':
    process_data()