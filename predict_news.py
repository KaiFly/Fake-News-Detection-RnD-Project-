
# This look like work on news articles in english and contain information about politics of USA

import pickle
from sentiment_score import sen_feature
from clean_sen import process_data
import scipy as sp
import numpy as np
with open('pac_model.sav', "rb") as infile:
    PAC_model = pickle.load(infile)
with open('tfidf.sav', "rb") as infile:
    tfidf_transform = pickle.load(infile)


sen_test = input(" >>News to test :")

"""sen_test = "In 2013, Clinton told Goldman Sachs bigwigs: \
        'I would like to see people like Donald Trump run for office.\
        They're honest, and can't be bought"
"""
sen_test = process_data(sen_test)
sentiment_score = sen_feature(sen_test)
X_sen = tfidf_transform.transform([sen_test])
X_sen = sp.sparse.hstack((X_sen, np.array([sentiment_score])),format='csr')
label_sen = PAC_model.predict(X_sen)
proba_truth_sen = PAC_model.decision_function(X_sen)[0]
proba_doubt_sen = (1 - abs(proba_truth_sen))*abs(proba_truth_sen)/(-proba_truth_sen)

print(" PAC_model :")
print(" This new is : " + label_sen[0])
print(" The sentiment score         :" + str(sentiment_score))
print(" The truth score             :" + str(proba_truth_sen))
print(" The doublt score            :" + str(proba_doubt_sen))