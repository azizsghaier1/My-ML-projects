import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('winequality-red.csv')
print(data.head())
# print(data.dtypes)
# print(data.describe())
# print(data.quality.value_counts())
# sns.countplot(x='quality',data=data)
# plt.figure(figsize=(5,5))
# sns.barplot(x='quality',y='citric acid', data=data)
# corre=data.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corre,cbar=True,square=True,fmt='.1f',annot=True,cmap='Blues')
# plt.show()
X=data.drop(columns='quality',axis=1)
y=data['quality']
data.replace({'quality':{3:0,4:0,5:0,6:0,7:1,8:1}},inplace=True)
print(data.quality.value_counts())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
model=RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)
# y_p=model.predict(X_test)
# acc=accuracy_score(y_test,y_p)
# print(acc)
#Input system
ask='yes'
while ask.upper()=='YES' :
    x=input('give me ur data')
    l=x.split(',')
    l=[float(i) for i in l]
    x=np.asarray(l)
    x=x.reshape(1,-1)
    prediction=model.predict(x)
    if prediction[0]==1 :
        print("it's a good wine ")
    else:
        print("it's bad  ")
    ask=input("Have u other datasets ?")

