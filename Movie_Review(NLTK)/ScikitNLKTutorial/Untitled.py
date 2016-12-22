from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load the data
dataset = load_files('MovieReview/txt_sentoken', shuffle=False)
print("n_samples: %d" % len(dataset.data))

#Split the data into test dataset and training dataset
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=100, max_df=0.75)
X_train_counts = count_vect.fit_transform(docs_train)

total_files = len(X_train_counts.toarray())
print('Total number of files:')
print(len(X_train_counts.toarray()))
total_numwords = len(X_train_counts.toarray())[1]
print('Total number of words:')
print(len(X_train_counts.toarray()[1]))
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.toarray()[1])
