


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()

x_train,x_test,y_train,y_test= train_test_split(iris.data,iris.target, random_state=1)


clf=LogisticRegression()
clf.fit(x_train,y_train)

y_train_pred=clf.predict(x_train)
l
y_test_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train,y_train_pred)

confusion_matrix(y_test,y_test_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_test_pred))
