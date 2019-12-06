from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.utils import shuffle

#Reading training data set
df = pd.read_csv('fordTrain.csv',sep=",",header=0,usecols=['IsAlert','P1','P2','P3','P4','P5','P6','P7','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','V1','V2','V3','V4','V5','V6','V8','V10','V11'],nrows=604330)

#Dividing it into X and y
X = np.array(df.drop(['IsAlert'],1))
y = np.array(df['IsAlert'])

#Preprocessing scales the feature values. Cross validation shuffles them and splits into train(80%) and test(20%) datasets
X = preprocessing.scale(X)
X_train,y_train = shuffle(X, y, random_state=0)
#X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


gnb = GaussianNB()
gnb.fit(X_train, y_train)


#testing data
X_test,y_test = shuffle(X, y, random_state=0)
accuracy = gnb.score(X_test,y_test)*100
error=100-accuracy
print("Accuracy = %f" %accuracy)
print("Train Error = %f" %error)

#False Positive Rate for Train Data
ans = gnb.predict(X_test)
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
X = np.array(pd.read_csv('fordTest.csv',sep=",",header=0,usecols=['P1','P2','P3','P4','P5','P6','P7','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','V1','V2','V3','V4','V5','V6','V8','V10','V11'],nrows=120841))
y = np.array(pd.read_csv('Solution.csv',sep=",",header=0,usecols=['Prediction'],nrows=120841))
             
X = preprocessing.scale(X)
accuracy = gnb.score(X,y)*100
error=100-accuracy
print("Accuracy = %f" %accuracy)
print("Test Error = %f" %error)

#False Positive Rate for Test Data
ans = gnb.predict(X)
count=0
total=0
for a,b in zip(ans,y):
    if (a!=b and a==1):
        count+=1
    if (a!=b):
        total+=1        
        
res=(count/total)*100
print("Test False Positive = %f" %res)





