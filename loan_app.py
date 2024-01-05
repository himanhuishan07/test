import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('Cleaned_loan.csv')
print(df.head())

df = df.drop('Unnamed: 0',axis=1)
print(df.head())

# Train_Test_split
x = df.drop('Loan_Status',axis=1)
y = df[['Loan_Status']]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,random_state=15,stratify=y)

# instantiate the model

log_reg = LogisticRegression()

# fit the model
log_reg.fit(x_train,y_train)

# make pickle file for our model

pickle.dump(log_reg,open('model.pkl','wb'))