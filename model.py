import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#loading the dataset 
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

#preprocessing the dataset
#removing ID and separate targeet feature
x=df.drop(columns=['customerID','Churn'])
y=df['Churn'].map({'Yes': 1, 'No': 0}) 
# converting target variable to binary  

# encode categorical features using one-hot encoding
x_encoded = pd.get_dummies(x, drop_first=True)

#strarified train_test split (id dataset is imbalanced)
x_train, x_test, y_train, y_test = train_test_split(x_encoded,y,test_size=0.2,random_state=42,stratify=y)

#features scaling of  the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#model training using random forest classifier
model=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(x_train_scaled,y_train)

#evaluation of the model
y_pred = model.predict(x_test_scaled)
print(f"model accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))