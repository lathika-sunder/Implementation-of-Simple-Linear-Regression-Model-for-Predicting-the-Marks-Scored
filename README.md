## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
### DATE:13.02.2024
### AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

### Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Lathika Sunder
RegisterNumber:  212221230054
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```
### Output:
### 1)HEAD:
![image](https://github.com/gpavana/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787343/b63f656e-500f-4008-ad94-472539f0f910)
### 2)GRAPH OF PLOTTED DATA:
![image](https://github.com/gpavana/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787343/c81e2c1b-901f-4649-ab92-c88901e26554)
### 3)TRAINED DATA:
![image](https://github.com/gpavana/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787343/72b1e994-e121-40c4-965a-d92c6283d9ec)
### 4)LINE OF REGRESSION:
![image](https://github.com/gpavana/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787343/dd31869f-4386-44c5-9877-db7321bac033)
### 5)COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/gpavana/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787343/882b225d-9ed3-431b-85a7-e94051165bec)

### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
