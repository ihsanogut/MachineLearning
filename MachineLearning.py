import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import graphviz
from sklearn.model_selection import train_test_split


df=pd.read_csv('heart.csv')
print(df.head(4))

print(df.info())

y=df['output']
print(y.head(4))

x=df.drop(columns="output")

print(x.head(4))

tree=DecisionTreeClassifier()
model=tree.fit(x,y)

print("full score "+str(model.score(x,y)))


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=16,train_size=0.7)

print("x_train ")
print(x_train)

print("y_train ")
print(y_train)

tree=DecisionTreeClassifier()
model=tree.fit(x_train,y_train)

print("test score "+str(model.score(x_test,y_test)))

input=[{67,1,0,160,286,0,0,108,1,1.5,1,3,2}]

#age=67,sex=1,cp=0,trtbps=160,chol=286,fbs=0,restecg=0,thalachh=108,exng=1,oldpeak=1.5,slp=1,caa=3,thall=2

#input=[{'age':67,'sex':1,'cp':0,'trtbps':160,'chol':286,'fbs':0,'restecg':0,'thalachh':108,'exng':1,'oldpeak':1.5,'slp':1,'caa':3,'thall':2}]

print("hhhhhhhhhhhhhhhhhhhhhhhhhh")
print(input)

predictions=model.predict(x,input)





