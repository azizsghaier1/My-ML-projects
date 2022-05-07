import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#data collectin and data processing
data=pd.read_csv('sonar.all-data.csv',header=None)
print(data[60].value_counts())
print(data.groupby(60).mean())

y=data[60]
X=data.drop(columns=60,axis=0)
print(X.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=1)
model=LogisticRegression()
print(X_train)
print(y_train)
model.fit(X_train,y_train)
#evaluation
y_p=model.predict(X_test)
print(accuracy_score(y_test,y_p))
#making a predective system
dar=(0.0201,0.0116,0.0123,0.0245,0.0547,0.0208,0.0891,0.0836,0.1335,0.1199,0.1742,0.1387,0.2042,0.2580,0.2616,0.2097,0.2532,0.3213,0.4327,0.4760,0.5328,0.6057,0.6696,0.7476,0.8930,0.9405,1.0000,0.9785,0.8473,0.7639,0.6701,0.4989,0.3718,0.2196,0.1416,0.2680,0.2630,0.3104,0.3392,0.2123,0.1170,0.2655,0.2203,0.1541,0.1464,0.1044,0.1225,0.0745,0.0490,0.0224,0.0032,0.0076,0.0045,0.0056,0.0075,0.0037,0.0045,0.0029,0.0008,0.0018)
input_data=np.asarray(dar)
input_reshaped=input_data.reshape(1,-1)
pred=model.predict(input_reshaped)
if pred=='M':
    print('its a Mine be careful !')
else:
    print("don't worry it's just a rock !")