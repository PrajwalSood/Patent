import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('train-data-final.csv')
y_train = np.array(data1['Label'])
X_train = data1.drop('Label', axis = 1)
X_train = X_train.drop('Publication Number', axis =1)

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
    

    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train['Title']).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_train ,y_train, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(xtrain, ytrain)

y_pred_gnb = classifier.predict(xtest)
acc_gnb = round(classifier.score(xtrain, ytrain) * 100, 2)
print (acc_gnb) #96.98

y_pred_gnb = le.inverse_transform(y_pred_gnb)
ytest = le.inverse_transform(ytest) 
import sklearn.metrics as m
cf_svc = m.confusion_matrix(ytest, y_pred_gnb)

report = pd.m.classification_report(ytest, y_pred_gnb)

m.f1_score(ytest, y_pred_gnb, average = None)


