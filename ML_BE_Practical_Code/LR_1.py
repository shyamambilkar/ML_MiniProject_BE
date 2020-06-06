import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Hours.csv")
x=dataset.iloc[:,:-1].values
x=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x)
acc=reg.score(x)*100
print(acc)

y_predict=reg.predict([[10]])
print(y_predict)
print(x)

plt.plot(x,x,'s')
plt.plot(reg.predict(x))
plt.show()



