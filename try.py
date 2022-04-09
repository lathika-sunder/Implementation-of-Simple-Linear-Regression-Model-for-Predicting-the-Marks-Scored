import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('/student_scores - student_scores.csv')
dataset.head()
x=dataset.iloc[:,:-1].values #iloc[:,start_column:end_column]
y=dataset.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('hours vs scores(training set)')
plt.x_label='hours'
plt.y_label=('scores')
plt.show