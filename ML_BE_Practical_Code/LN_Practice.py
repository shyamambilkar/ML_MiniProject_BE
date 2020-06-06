import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("hours.csv")
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