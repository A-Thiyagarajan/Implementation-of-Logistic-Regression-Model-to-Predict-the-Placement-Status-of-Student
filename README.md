# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 

3.Import LabelEncoder and encode the dataset. 

4.Import LogisticRegression from sklearn and apply the model on the dataset. 

5.Predict the values of array. 

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 212222240110
RegisterNumber:  Thiyagarajan A

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:

### Placement Data: 

![placement](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/d0ccab75-256f-4f14-b1e1-3f93cc0ae92a)


### Salary Data:

![salary](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/d9982d36-38e7-44dc-924b-15f8d96a5fff)


### Checking the null() function:

![check](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/75b24d78-a5f7-4269-8214-274cc416a0de)



### Data Duplicate:


![dul](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/d89cf75f-e006-4e92-b827-9e7442f8523e)


### Print Data:


![print](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/f2ff24f1-99d4-41d0-8687-ea720bd1f0aa)


### Data-status:

![status](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/ba928a71-4c5f-4350-b12d-80e05ba607ff)


### y_prediction array:

![y-arr](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/eec92ff3-0571-405e-ae6e-6c2355348f5b)



### Accuracy value:

![acc](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/2a0e0ffd-588a-4c76-a63f-53f4013fb05e)



### Confusion array:


![con-arr](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/c6d6239f-991f-4042-8a53-7b733d681466)



### Classification report:


![report](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/9dc02e02-90d2-4bd1-b8d9-7a3f5beee06a)


### Prediction of LR:

![lr](https://github.com/A-Thiyagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707693/01702721-a55a-43ad-b76f-8a85d22902e6)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
