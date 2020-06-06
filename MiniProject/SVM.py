import pandas as pd  
import numpy as np  
#import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  

from sklearn import svm  
from sklearn.metrics import classification_report, confusion_matrix  

print("hi!!!!!!Welcome to our system")
from sklearn.tree import DecisionTreeClassifier  
#'exec(%matplotlib inline)'

train = pd.read_csv("Social_Network_Ads.csv")
#print(train.shape)

#print(train.head())

X= train.drop('label', axis=1)  
Y = train['label']  


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20) 


#svclassifier = SVC(kernel='linear')  
#svclassifier.fit(X_train,y_train)  

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)  

print(X_test)
 



y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred)) 


print(classification_report(y_test,y_pred)) 


 
