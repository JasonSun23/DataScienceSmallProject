import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from operator import truediv

#Load the data
dataset = load_files('MovieReview/txt_sentoken', shuffle=False)
print("n_samples: %d" % len(dataset.data))


#Split the data into test dataset and training dataset
#docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)
print(len(dataset.target))


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=300, max_df=0.95, stop_words='english', ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(dataset.data)
features = count_vect.get_feature_names()
#print(features)

total_num_words = len(X_train_counts.toarray()[1])
total_num_files = len(X_train_counts.toarray())
print('Total number of files:')
print(len(X_train_counts.toarray()))
print('Total number of words:')
print(len(X_train_counts.toarray()[1]))


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.toarray()[1][3])

sum_words_tfidf = [0] * total_num_words
sum_words_sentiment = [0] * total_num_words
n = [0] * total_num_words

for j in range(total_num_files):
    print('Working on...')
    print(j+1)
    for i in range(total_num_words):
        sum_words_tfidf[i] = X_train_tfidf.toarray()[j][i] + sum_words_tfidf[i]
        if X_train_tfidf.toarray()[j][i] > 0:
            n[i] = n[i] + 1
            sum_words_sentiment[i] = dataset.target[j] + sum_words_sentiment[i]

y = [num / total_num_files for num in sum_words_tfidf]*10
print(y)

x = [0] * total_num_words
for i in range(len(n)):
    if n[i] != 0:
        x[i] = (x[i] + (sum_words_sentiment[i]/n[i]))
print(x)

size = n


import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('JasonSun23', 'ze2jekbjzj')
data = [
    {
        'x': x,
        'y': y,
        'text': features,
        'mode': 'markers',
        'marker': {
            'color': x,
            'size': size,
            'sizeref': 40,
            'showscale': True
        }
    }
]

py.iplot(data, filename='scatter-colorscale')
