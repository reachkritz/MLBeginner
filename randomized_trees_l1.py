from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing,cross_validation,metrics
import pandas as ps
import numpy as np
from sklearn.utils import shuffle

df=ps.read_csv('fordTrain.csv',sep=",",header=0,nrows=350000,usecols=['IsAlert','P1','P2','P3','P4','P5','P6','P7','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','V1','V2','V3','V4','V5','V6','V8','V10','V11'])

#print attributes
#print(df.columns)

#separate features from labels
X = np.array(df.drop(['IsAlert'],1))
y = np.array(df['IsAlert'])

#Preprocessing scales the feature values. Cross validation shuffles them and splits into train(80%) and test(20%) datasets
X = preprocessing.scale(X)
X_train,y_train = shuffle(X, y, random_state=0)
#X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
            
clf = RandomForestClassifier()
clf.fit(X_train,y_train)

X_test,y_test = shuffle(X, y, random_state=0)
accuracy=clf.score(X_test,y_test)
print("accuracy:",accuracy*100)
y_predict=clf.predict(X_test)

performance_matrix=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])
print("Train error",(1-accuracy)*100)
total=(performance_matrix[0][1]+performance_matrix[1][0])
count=performance_matrix[0][1]
print("Train False Positive : ",(count/total)*100)

#Loading Actual test Data
df=ps.read_csv('fordTest.csv',sep=",",usecols=['IsAlert','P1','P2','P3','P4','P5','P6','P7','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','V1','V2','V3','V4','V5','V6','V8','V10','V11'],header=0)
X = np.array(df.drop(['IsAlert'],1))
X = preprocessing.scale(X)
y_predict=clf.predict(X)

y_true = np.array(ps.read_csv('solution.csv',sep=",",header=0,usecols=['Prediction']))
accuracy = clf.score(X,y_true)
print("accuracy:",accuracy*100)
#print(accuracy)

performance_matrix=metrics.confusion_matrix(y_true,y_predict,labels=[0,1])             
print("Test Error : ",(1-accuracy)*100)
total=(performance_matrix[0][1]+performance_matrix[1][0])
count=performance_matrix[0][1]
print("test False Positive : ",(count/total)*100)


