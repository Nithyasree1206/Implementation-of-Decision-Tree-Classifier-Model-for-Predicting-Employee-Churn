## Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Developed by: NITHYASREE S
## RegisterNumber:212224040225

```PYTHON
import matplotlib.pyplot as plt    
%matplotlib inline                  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv(r"C:\Users\L390 Yoga\Downloads\Employee.csv")
data.head()

data.info()

data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left

y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(15,10))
plot_tree(dt,
          feature_names=x.columns,
          class_names=['stayed','left'],
          filled=True)

plt.show()

```
## Output:
<img width="659" height="182" alt="image" src="https://github.com/user-attachments/assets/961f7d78-13f6-48fa-b1f9-ba5b0ebed02c" />
<img width="1290" height="373" alt="image" src="https://github.com/user-attachments/assets/f4816b99-eaac-4eda-ba7c-c3784718429b" />
<img width="1285" height="267" alt="image" src="https://github.com/user-attachments/assets/3f3da32d-a42e-472a-80dc-23e6d26b03f1" />
<img width="1282" height="142" alt="image" src="https://github.com/user-attachments/assets/d4c91064-d362-4fa0-8602-b662c14322fe" />
<img width="1290" height="233" alt="image" src="https://github.com/user-attachments/assets/b6aedac1-40d9-4aa8-abbc-e9029e693803" />
<img width="614" height="151" alt="image" src="https://github.com/user-attachments/assets/11253ac1-4674-432f-b33d-f010be969004" />

<img width="1112" height="708" alt="image" src="https://github.com/user-attachments/assets/88c4ffd2-a93a-46cf-9243-c168971fa370" />


## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
