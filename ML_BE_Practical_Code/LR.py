import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Hours.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)
acc=reg.score(x,y)*100
print(acc)

y_predict=reg.predict([[10]])
print(y_predict)
print(y)

plt.plot(x,y,'s')
plt.plot(reg.predict(x))
plt.show()

