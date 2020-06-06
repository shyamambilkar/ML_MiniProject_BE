import pandas as pd
import numpy as np

dataset=pd.read_csv("kdata.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x,y)

x_test=np.array([6,6])
y_pred=classifier.predict([x_test])
print('General KNN',y_pred)

classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(x,y)

x_test=np.array([6,2])
y_pred=classifier.predict([x_test])
print('Distance weight KNN',y_pred)