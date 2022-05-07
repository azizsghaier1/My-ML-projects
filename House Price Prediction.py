import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,accuracy_score
from xgboost import XGBRegressor

data = sklearn.datasets.load_boston()
dataf = pd.DataFrame(data['data'], columns=data['feature_names'])
dataf['price'] = data.target
# print(dataf.head())
# check for missing values
# print(dataf.isnull().sum())
# statistical info
# print(dataf.describe())
# correlation
# corr = dataf.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Blues')
# plt.show()
X=dataf.drop('price',axis=1)
y=dataf['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
model=XGBRegressor(n_estimator=200)
model.fit(X_train,y_train)
y_p=model.predict(X_test)
# print(y_p.shape,'--',y_test.shape)
# print(y_p )
error=mean_squared_error(y_test,y_p)
# print("error=",error)
# sns.histplot(data=dataf,x=y_test,y=y_p)
# plt.xlabel("real price ")
# plt.ylabel("predicted price")
# plt.title('figure')
# plt.show()
#input data :
ch=input('give me ur data ')
l=ch.split(",")
l=[float(i) for i in l ]
ar=np.asarray(l)
ar=ar.reshape(1,-1)
price=model.predict(ar)
print('The price of this house is :{}'.format(price))
