# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SUROTHAAMAN R
RegisterNumber: 212222103003
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
```
import pandas as pd
data=pd.read_csv("/content/Salary-2.csv")
data.head()
```
<img width="325" alt="Screenshot 2024-04-01 at 9 48 36 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/996f72fe-1e0f-4216-8710-a196862a0b72">

```
data.info()
```
<img width="363" alt="Screenshot 2024-04-01 at 9 48 43 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/5123640f-99f3-4565-9c57-9d5c744d29f8">

```
data.isnull().sum()
```
<img width="156" alt="Screenshot 2024-04-01 at 9 48 49 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/9cdb650d-928f-4139-89e2-bda4a4ff1263">

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
<img width="288" alt="Screenshot 2024-04-01 at 9 48 56 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/657e5e09-fcd9-4b9f-8990-83ebc80f2bef">

```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
```
<img width="122" alt="Screenshot 2024-04-01 at 9 49 09 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/d16bc1b2-e69c-4a9a-93ac-46c640173b8d">

```
r2=metrics.r2_score(y_test,y_pred)
r2
```
<img width="179" alt="Screenshot 2024-04-01 at 9 49 16 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/98ca15c0-9751-4cca-9a1c-16bdda358017">


```
dt.predict([[5,6]])
```
<img width="1354" alt="Screenshot 2024-04-02 at 11 07 51 AM" src="https://github.com/Richard01072002/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141472248/68310f6c-8527-4496-809e-df64a33eff23">








## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
