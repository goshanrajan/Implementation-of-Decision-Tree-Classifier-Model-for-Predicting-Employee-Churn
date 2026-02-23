# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the Employee dataset.

2.Separate features (X) and target variable (y), then split into training and testing sets.

3.Train the Decision Tree Classifier model and calculate accuracy.

4.Get user input data and predict whether the employee will churn or not. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: T.Goshanrajan
RegisterNumber:  212225040098

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("C:/Users/acer/Downloads/Employee.csv")

# Automatically select last column as target
target_column = data.columns[-1]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

print("Decision Tree Classifier Model")
print("Accuracy:", accuracy_score(y_test, y_pred))

# User Input Section
print("\nEnter values to predict employee churn:")

user_input = []
for col in X.columns:
    value = float(input(f"Enter value for {col}: "))
    user_input.append(value)

user_df = pd.DataFrame([user_input], columns=X.columns)

prediction = model.predict(user_df)

if prediction[0] == 1:
    print("Employee is likely to Churn")
else:
    print("Employee is not likely to Churn")
```

## Output:
<img width="811" height="496" alt="image" src="https://github.com/user-attachments/assets/ea01dfcb-5752-4306-9dad-57a5539f9c87" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
