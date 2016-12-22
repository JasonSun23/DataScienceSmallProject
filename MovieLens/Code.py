import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as py

#Q1

unames = ['user_id', 'gender', 'age', 'occupation', 'zipc']
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python')

'''
print(users[:5])
print(ratings[:5])
print(movies[:5])
print(ratings)
'''

data = pd.merge(pd.merge(ratings, users), movies)

'''
print(data[:5])
print(data.ix[1])
'''

store = pd.HDFStore('data.h5')
store['data'] = data

'''
print(store['data'])
'''

#

mean_ratings = data.pivot_table('rating', index='title', aggfunc='mean')
active_titles = mean_ratings.index[mean_ratings > 4.5]
mean_rating_fpf = mean_ratings.ix[active_titles]
#print(mean_rating_fpf)
print('Rating over 4.5:', len(mean_rating_fpf))

#

mean_ratings_gender = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
active_titlesM = mean_ratings_gender.index[mean_ratings_gender['M'] > 4.5]
active_titlesF = mean_ratings_gender.index[mean_ratings_gender['F'] > 4.5]

mean_ratings_genderM_fpf = mean_ratings_gender.ix[active_titlesM]
mean_ratings_genderF_fpf = mean_ratings_gender.ix[active_titlesF]
#print(mean_ratings_genderM_fpf['M'])
#print(mean_ratings_genderF_fpf['F'])
print('Rating over 4.5 among Female:', len(mean_ratings_genderF_fpf))
print('Rating over 4.5 among Male:', len(mean_ratings_genderM_fpf))

#

data_over_age = data[data.age > 30]
mean_ratings_gender_age = data_over_age.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
active_titlesM = mean_ratings_gender_age.index[mean_ratings_gender_age['M'] > 4.5]
active_titlesF = mean_ratings_gender_age.index[mean_ratings_gender_age['F'] > 4.5]

mean_ratings_genderM_age_fpf = mean_ratings_gender_age.ix[active_titlesM]
mean_ratings_genderF_age_fpf = mean_ratings_gender_age.ix[active_titlesF]
#print(mean_ratings_genderM_fpf['M'])
#print(mean_ratings_genderF_fpf['F'])
print('Rating over 4.5 among Female over 30:', len(mean_ratings_genderF_age_fpf))
print('Rating over 4.5 among Male over 30:', len(mean_ratings_genderM_age_fpf))

#

popularity = data.groupby('title').size()
print(popularity.sort_values(ascending=False)[0:1])

#Delet those movies under 250 ratings??

occu= data.pivot_table('rating', index='title', columns='occupation', aggfunc='mean')
occu = occu.dropna()
result = pd.concat([occu.mean(axis=0), occu.std(axis=0)], axis=1)
result.columns = ['Mean', 'Std']
print(result.sort_values(by='Mean', ascending=False))

#

gen = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
gen = gen.dropna()
result = pd.concat([gen.mean(axis=0), gen.std(axis=0)], axis=1)
result.columns = ['Mean', 'Std']
print(result.sort_values(by='Mean', ascending=False))

#

age = data.pivot_table('rating', index='title', columns='age', aggfunc='mean')
age = age.dropna()
result = pd.concat([age.mean(axis=0), age.std(axis=0)], axis=1)
result.columns = ['Mean', 'Std']
print(result.sort_values(by='Mean', ascending=False))


#Q2

plt.hist(data['rating'], bins=20)
plt.title('Histogram of the ratings of all movies')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

#

NumOfRat = data.pivot_table('rating', index='title', aggfunc='count')
plt.hist(NumOfRat, bins=20)
plt.title('Histogram of the number of ratings')
plt.xlabel('Number of ratings')
plt.ylabel('Frequency')
plt.show()

#

AvgOfRat = data.pivot_table('rating', index='title', aggfunc='mean')
plt.hist(AvgOfRat, bins=20)
plt.title('Histogram of the average rating')
plt.xlabel('Average rating')
plt.ylabel('Frequency')
plt.show()

#

active_titles = popularity.index[popularity >= 100]
AvgOfRatAbv = AvgOfRat.ix[active_titles]
plt.hist(AvgOfRatAbv, bins=20)
plt.title('Histogram of the average rating for movies with more than 100 rating times')
plt.xlabel('Average rating')
plt.ylabel('Frequency')
plt.show()


for i in range(21):
    mean = occu[i].mean(axis=0)
    std = occu[i].std(axis=0)
    py.subplot(5, 5, i+1)
    plt.hist(occu[i])
    plt.xlabel('Rating', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.title(r'Occupation: %d $\mu=%2f,\ \sigma=%2f$' %(i, mean, std), fontsize=8)

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, left=0.1)
plt.show()

j=1
for i in ['M', 'F']:
    mean = gen[i].mean(axis=0)
    std = gen[i].std(axis=0)
    py.subplot(1, 2, j)
    plt.hist(gen[i])
    plt.xlabel('Rating', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.title(r'Gender: %s $\mu=%2f,\ \sigma=%2f$' %(i, mean, std), fontsize=8)
    j = j+1
plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, left=0.1)
plt.show()

j=1
for i in [1, 18, 25, 35, 45, 50, 56]:
    mean = age[i].mean(axis=0)
    std = age[i].std(axis=0)
    py.subplot(3, 3, j)
    plt.hist(age[i])
    plt.xlabel('Rating', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.title(r'Age: %s $\mu=%2f,\ \sigma=%2f$' %(i, mean, std), fontsize=8)
    j = j+1
plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9, left=0.1)
plt.show()

#Q3

py.plot(gen['M'],gen['F'],'.')
plt.show()

#

active_titles = popularity.index[popularity >= 200]
genAbv = gen.ix[active_titles]
py.plot(genAbv['M'],genAbv['F'],'.')
plt.show()

#

print('Correlation:', gen.corr())
print('Correlation with movie over 200 ratings:', genAbv.corr())

#

genVSage = data.pivot_table('rating', index='title', columns=['age', 'gender'], aggfunc='mean')
j = 1
for i in [1, 18, 25, 35, 45, 50, 56]:
    py.subplot(3, 3, j)
    py.plot(genVSage[i]['M'], genVSage[i]['F'], '.')
    py.title('Age: %d' %(i))
    j = j + 1
    print('Correlation of age', i)
    print(genVSage[i].corr())
    print('--------------------------')

plt.show()
