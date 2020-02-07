import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

classifier = joblib.load('my_model.pkl')
cv = joblib.load('cv.pkl')
le = joblib.load('le.pkl')

from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
from pandas import DataFrame




pub_file= input()
print(pub_file)


driver=webdriver.Firefox(executable_path ='geckodriver.exe')

driver.get('https://patents.google.com/')
time.sleep(3)

wait = WebDriverWait(driver, 10)

pp=[]
for i in range(0,2):
	
	driver.find_element_by_id('searchInput').clear()
	driver.find_element_by_id('searchInput').send_keys(pub_file)
	time.sleep(5)

	driver.find_element_by_id('searchButton').click()
	time.sleep(7)
	
	pub_name=driver.find_element_by_id('title').text
	print(pub_name)
	
	time.sleep(7)

driver.quit()


review = pub_name
review = re.sub('[^a-zA-Z]',' ',str(pub_name))
review = review.lower()
review = review.split()
    
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
review = ' '.join(review)
review = [review,]

review = cv.transform(review).toarray()
y_pred = classifier.predict(review)

y_pred = le.inverse_transform(y_pred)
y_pred = y_pred[0]
