from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.utils import shuffle

#Reading training data set
df = pd.read_csv('fordTrain.csv',sep=",",header=0,usecols=['IsAlert','E8','E9','V11'],nrows=350000)

#Dividing it into X and y
X = np.array(df.drop(['IsAlert'],1))
y = np.array(df['IsAlert'])

#Preprocessing scales the feature values. Cross validation shuffles them and splits into train(80%) and test(20%) datasets
X = preprocessing.scale(X)
#X_train,y_train = shuffle(X, y, random_state=0)
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(30, 100),random_state=1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)*100
error=100-accuracy
print("Accuracy %f" %accuracy)
print("Train Error %f" %error)

#False Positive Rate for Train Data
ans = clf.predict(X_test)
count=0
total=0
for a,b in zip(ans,y_test):
    if (a!=b and a==1):
        count+=1
    if (a!=b):
        total+=1        
        
res=(count/total)*100
print("Train False Positive = %f" %res)

#Loading Actual test Data
X = np.array(pd.read_csv('fordTest.csv',sep=",",header=0,usecols=['E8','E9','V11']))
y = np.array(pd.read_csv('Solution.csv',sep=",",header=0,usecols=['Prediction']))
             
X = preprocessing.scale(X)
accuracy = clf.score(X,y)*100
error=100-accuracy
print("Accuracy %f" %accuracy)
print("Test Error %f" %error)

#False Positive Rate for Test Data
ans = clf.predict(X)
count=0
total=0
for a,b in zip(ans,y):
    if (a!=b and a==1):
        count+=1
    if (a!=b):
        total+=1        
        
res=(count/total)*100
print("Test False Positive = %f" %res)
