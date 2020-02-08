import turtle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import re
import pyglet
from sklearn.externals import joblib
from playsound import playsound

def fun(a):
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
    
    
    
    
    pub_file= a
    
    
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
    return y_pred

screen = turtle.Screen()
#screen.bgcolor("black")
screen.setup(512,334)
screen.bgpic("bg.gif")
pen = turtle.Turtle()
pen.hideturtle()
#pen.reset()
#pen.color("blue")
#pen.setposition(-159, -67)

#for i in range(2): #box drawing
#    pen.forward(25)
#    pen.left(90)
#    pen.forward(57)
#    pen.left(90)
pen.hideturtle()
pen.penup()
pen.goto(-159, -50)
pen.color("white")
pen.write("   Click  \nto proceed", font=("Arial",12,"normal"))

def btnclick(x,y):
    print(x,y) #prints coordinates of click
    if x >-162  and x<-22 and y>-66 and y<0:
        
        answer = screen.textinput("Please enter patent number"," ")
        playsound("pleasewait.wav")
    if answer!=None:
        answer = fun(answer)
        window = pyglet.window.Window()
        label = pyglet.text.Label(answer,
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')
        @window.event
        def on_draw():
            window.clear()
            label.draw()
        pyglet.app.run()

turtle.onscreenclick(btnclick, 1)
turtle.listen()

turtle.done()


