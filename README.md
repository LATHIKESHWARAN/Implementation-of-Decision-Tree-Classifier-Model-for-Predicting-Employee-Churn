# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data
Clean and format your data
Split your data into training and testing sets

2.Define your model
Use a sigmoid function to map inputs to outputs
Initialize weights and bias terms

3.Define your cost function
Use binary cross-entropy loss function
Penalize the model for incorrect predictions

4.Define your learning rate
Determines how quickly weights are updated during gradient descent

5.Train your model
Adjust weights and bias terms using gradient descent
Iterate until convergence or for a fixed number of iterations

6.Evaluate your model
Test performance on testing data
Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters
Experiment with different learning rates and regularization techniques

8.Deploy your model
Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: LATHIKESHWARAN J
RegisterNumber:  212222230072
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

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
```

## Output:
### Initial data set:
![O1](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/c5159af9-1a7f-4dec-bc75-dc6224080535)
### Data info:
![O2](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/ee9d32ca-52ce-41ac-918d-b73f36dc7983)
### Optimization of null values:
![O3](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/e8da66f6-c556-4682-ad6d-58d610377bb4)
### Assignment of x and y values:
![O4](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/ef8abea2-028a-470e-888d-ba443b6a7e37)
![O5](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/116a1b66-3f8c-41bc-b1b0-361cc8c648d7)
### Converting string literals to numerical values using label encoder:
![O6](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/a3ddaaac-ec58-4db2-bec9-15877cd32822)
### Accuracy:
![O7](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/991fd46f-8339-4115-a68a-779393aa03bb)
### Prediction:
![O8](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393556/ab22a401-8fc6-44b7-9cd6-69e571755c27)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
