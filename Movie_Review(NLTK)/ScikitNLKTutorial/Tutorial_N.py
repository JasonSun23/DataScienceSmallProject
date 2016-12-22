categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset = 'train',categories = categories,shuffle = True,random_state = 42)
twenty_train.target_names

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

'''
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))
'''

from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
             'clf__alpha': (1e-2, 1e-3), }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])
