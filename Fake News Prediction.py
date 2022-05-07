import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')
l=stopwords.words('english')
#Data preprocessing
data=pd.read_csv('train fake or real.csv',index_col=0)
print(data.columns)
print(data.isnull().sum())
#PREPROCESSING WITH MISSING DATA
data=data.fillna('')
data['content']=data['author']+' '+data['title']

y=data['label']
port_stem=PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
data['content'] = data['content'].apply(stemming)
X=data['content']
#converting textual values to numerical values
vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)
print('X=',X)
#spliting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2,stratify=y)
model=LogisticRegression()
model.fit(X_train,y_train) #Y=sigmoid(Z) and Z=wX+b
# y_train_p=model.predict(X_train)
# y_test_p=model.predict(X_test)
# training_acc=accuracy_score(y_train,y_train_p)
# testing_acc=accuracy_score(y_test,y_test_p)
# print('accuracy score for training data = ',training_acc)
# print('accuracy score for testing data= ',testing_acc)
#input data
ask='no'
while ask.upper()=='NO':
    inp=input('give me the content of the newspaper')
    cont=stemming(inp)
    content=list(cont)
    content=np.asarray(content)
    content.reshape(1,-1)
    cont=vectorizer.transform(content)
    yp=model.predict(cont)
    if yp[0]==0 :
        print('the news is real')
    else:
        print("it's fake")
    ask=input('Do you have others contents ?')
