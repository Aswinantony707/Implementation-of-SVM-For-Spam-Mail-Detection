# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Loading and Preparation:
Detect file encoding using chardet and load the CSV file with the correct
encoding using pandas .
Check for missing values and extract features ( v1 ) and labels ( v2 ) from the
dataset.
2. Data Splitting:
Split the data into training and testing sets using train_test_split .
3. Feature Extraction:
 Convert text data into numerical features using CountVectorizer .
4. Model Training and Evaluation:
Train an SVM model ( SVC ) on the transformed training data and predict
results on the test set.
Evaluate the model's performance using accuracy score from
sklearn.metrics .

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ASWIN ANTONY.S
RegisterNumber:  212224220014
*/


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### data.head()
![Screenshot 2023-11-02 000307](https://github.com/Roselinjovita/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104296/f25a1488-2cf4-401c-b9a1-c718528f7009)


### data.info()

![Screenshot 2023-11-02 000315](https://github.com/Roselinjovita/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104296/701258ca-4918-4d2f-b242-e292c007eb6b)


### data.isnull().sum()


![Screenshot 2023-11-02 000327](https://github.com/Roselinjovita/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104296/f630a7bd-8b5b-4a6b-9339-efc41656d1a6)

### y_prediction value

![Screenshot 2023-11-02 000336](https://github.com/Roselinjovita/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104296/9f5f562a-6191-47cd-a533-57677ea21f50)

### Accuracy value
![Screenshot 2023-11-02 000342](https://github.com/Roselinjovita/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104296/22fc6d5f-3a59-4a8f-bf6e-5cc76de836a0)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.


