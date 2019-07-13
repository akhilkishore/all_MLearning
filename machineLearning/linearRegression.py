import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

#create inputs and output labels

X =  20 * np.random.rand(100,1)
Y = 4 + 3*X + 5*np.random.rand(100,1)

#split data set
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.5,random_state=1) 

model = LinearRegression()

model.fit(X_train,y_train)

print(model.intercept_)
print(model.coef_)


print(model.score(X_test,y_test))
