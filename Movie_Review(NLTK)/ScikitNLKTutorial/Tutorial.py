categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset = 'train',categories = categories,shuffle = True,random_state = 42)
twenty_train.target_names
#print(len(twenty_train.data))

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
#print(X_train_counts.shape)
#print(type(X_train_counts))
#print(count_vect.get_feature_names())
#print(X_train_counts.toarray())


#print(count_vect.vocabulary_.get(u'sdfsar'))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.toarray()[1])

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)


docs_new = ['sorry', 'drug is so good']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
