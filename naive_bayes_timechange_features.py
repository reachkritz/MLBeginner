from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing,cross_validation,metrics
import pandas as ps
import numpy as np
from sklearn.utils import shuffle

df = ps.read_csv('out_fordtrain.csv',sep=",",header=0,nrows=355000,usecols=['IsAlert','P5','E3','E5','E7','E8','E9','V2','V5','V10','MaxE1','MaxE2','MaxE3','MaxE8','MaxE9','MaxV6','MaxV8','MinE6','MinE9','MinV10','MinV8','RanE2','RanE5','RanE7','RanE8','RanV3'])
#print attributes
#print(df.columns)

#separate features from labels
X = np.array(df.drop(['IsAlert'],1))
y = np.array(df['IsAlert'])

#Preprocessing scales the feature values. Cross validation shuffles them and splits into train(80%) and test(20%) datasets
X = preprocessing.scale(X)
X_train,y_train = shuffle(X, y, random_state=0)
#X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
            
clf = GaussianNB()
clf.fit(X_train,y_train)

X_test,y_test = shuffle(X, y, random_state=0)
accuracy=clf.score(X_test,y_test)
#print(accuracy)
y_predict=clf.predict(X_test)

performance_matrix=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])
print("Train error",(1-accuracy)*100)
total=(performance_matrix[0][1]+performance_matrix[1][0])
count=performance_matrix[0][1]
print("Train False Positive : ",(count/total)*100)
#print(performance_matrix)
#print(performance_matrix[1][0],'',performance_matrix[0][1])

#Loading Actual test Data
df = ps.read_csv('out_fordtest.csv',sep=",",header=0,usecols=['IsAlert','P5','E3','E5','E7','E8','E9','V2','V5','V10','MaxE1','MaxE2','MaxE3','MaxE8','MaxE9','MaxV6','MaxV8','MinE6','MinE9','MinV10','MinV8','RanE2','RanE5','RanE7','RanE8','RanV3'])
X = np.array(df.drop(['IsAlert'],1))
X = preprocessing.scale(X)
y_predict=clf.predict(X)

#print(y_predict.size)
y_true = np.array(ps.read_csv('solution.csv',sep=",",header=0,usecols=['Prediction']))
accuracy = clf.score(X,y_true)
#print(accuracy)

#print(y_true.size)
performance_matrix=metrics.confusion_matrix(y_true,y_predict,labels=[0,1])             
print("Test Error : ",(1-accuracy)*100)

total=(performance_matrix[0][1]+performance_matrix[1][0])
count=performance_matrix[0][1]
#print(performance_matrix)
#print(performance_matrix[1][0],'',performance_matrix[0][1])
print("test False Positive : ",(count/total)*100)



