import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
# print(data.head())
# print(data.dtypes)
data=data.dropna()
# print(data.isnull().sum())
def repl(row):
    if row=='Y':
        return 1
    else:
        return 0
data['Loan_Status']=data['Loan_Status'].map(repl)
# print(data['Loan_Status'])
# data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
# print(data['Dependents'].value_counts())
data.replace({"Dependents":{'3+':'4'}},inplace=True)
# sns.countplot(x=data['Education'],hue='Loan_Status',data=data)
# sns.countplot(x='Married',hue='Loan_Status',data=data)
data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
X=data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y=data['Loan_Status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2,stratify=y)
print(X_train.shape,X_test.shape)
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)
y_p=classifier.predict(X_test)
y_p_train=classifier.predict(X_train)
t_acc=accuracy_score(y_test,y_p)
tr_acc=accuracy_score(y_train,y_p_train)
print('accurac for test ={} \n accuracy for train= {}'.format(t_acc,tr_acc))