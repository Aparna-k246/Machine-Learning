
from sklearn import datasets

boston=datasets.load_boston()
X=boston.data
Y=boston.target

X.shape

import pandas as pd
df=pd.DataFrame(X)
print(boston.feature_names)
df.columns=boston.feature_names
df["age_age"] = df.AGE ** 2
df.describe()
X2= df.values
X2.shape

from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,random_state=0)
X2_train,X2_test,Y2_train,Y2_test=model_selection.train_test_split(X2,Y,random_state=0)

from sklearn.linear_model import LinearRegression
alg1=LinearRegression()
alg2=LinearRegression()

alg1.fit(X_train,Y_train)
alg2.fit(X2_train,Y2_train)

Y_pred=alg1.predict(X_test)
train_score=alg1.score(X_train,Y_train)
test_score=alg1.score(X_test,Y_test)
print("Train Score: ",train_score)
print("Test Score: ",test_score)

train2_score=alg2.score(X2_train,Y2_train)
test2_score=alg2.score(X2_test,Y2_test)
print("Train2 Score: ",train2_score)
print("Test2 Score: ",test2_score)