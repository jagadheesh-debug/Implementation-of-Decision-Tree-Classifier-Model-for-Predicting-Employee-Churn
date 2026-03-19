# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load employee data and split it into training and testing sets.

2.Train a Decision Tree classifier using entropy as the split criterion.

3.Evaluate the model using accuracy, confusion matrix, and classification report.

4.Use the trained model to predict whether a new employee will stay or leave.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:jagadheesh kumar T
RegisterNumber:212225040139
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy",random_state=100)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
sample=[[0.5,0.8,9,260,6,0,1,2]]
print("Prediction for sample:",dt.predict(sample))
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=x.columns,class_names=["stayed","left"],filled=True,rounded=True,fontsize=10)
plt.show()
```

## Output:
![WhatsApp Image 2026-03-19 at 11 10 01 AM](https://github.com/user-attachments/assets/834ec52c-68a6-466e-9aca-5fafd75f34e1)
![WhatsApp Image 2026-03-19 at 11 10 01 AM](https://github.com/user-attachments/assets/d6feba4f-b9c3-47d7-a656-89c410642eee)


![WhatsApp Image 2026-03-19 at 11 10 47 AM](https://github.com/user-attachments/assets/69bdc252-8e53-41d0-9b3f-72146d9777d6)

![WhatsApp Image 2026-03-19 at 11 11 07 AM](https://github.com/user-attachments/assets/56314ee2-ead3-4a12-b454-ccf592ef4119)
![WhatsApp Image 2026-03-19 at 11 11 29 AM](https://github.com/user-attachments/assets/880c1f12-577a-461c-ad28-1c4112017291)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
