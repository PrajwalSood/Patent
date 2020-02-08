import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('train-data-final.csv')
y_train = np.array(data1['Label'])
X_train = data1.drop('Label', axis = 1)
X_train = X_train.drop('Publication Number', axis =1)
X_test = pd.read_csv('predictions.csv')
X_test = X_test['Title']
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1200):
    review = re.sub('[^a-zA-Z]',' ',str(X_train['Title'][i]))
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    X_train['Title'][i]=review
    
for i in range(200):
    review = re.sub('[^a-zA-Z]',' ',str(X_test['Title'][i]))
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    X_test['Title'][i]=review

    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train['Title']).toarray()
X_test = cv.transform(X_test['Title']).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)





y_pred = list(classifier.predict(X_test))

for i in range(200):
    y_pred = le.inverse_transform(y_pred)




