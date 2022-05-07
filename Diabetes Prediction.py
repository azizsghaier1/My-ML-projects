import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#dataa collection and analysis
data=pd.read_csv('diabetes.csv')
print(data.head())
y=data['Outcome']
X=data.drop(columns='Outcome',axis=1)
print(data.groupby('Outcome').mean())
#data standarization --->we have a different ranges
scaler=StandardScaler()
scaler.fit(X)
standard_data=scaler.transform((X))
print(standard_data)
X=standard_data
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=2,stratify=y)
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)
#Evaluation :
# y_p=classifier.predict(X_test)
# accvalidaton=accuracy_score(y_test,y_p)
# acctest=accuracy_score(y_train,classifier.predict(X_train))
# print("accuracy test = {} otherwises accuracy train={}".format(accvalidaton,acctest))
#input data
ch=input("give us ur data :")
l=ch.split(",")
l=[float(i) for i in l ]
ar=np.asarray(l)
ar=ar.reshape(1,-1)
standard_data=scaler.transform(ar)
outp=classifier.predict(standard_data)
if outp==1 :
    print('unfortnabely you are diabetic')
else:
    print('you are not diabetic')