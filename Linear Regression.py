 
import numpy as np
data =np.loadtxt("/LR data.csv", delimiter=',')
x=data[:,0]
y=data[:,1]
x.shape

from sklearn import model_selection
X_train, X_test, Y_train, Y_test=model_selection.train_test_split(x,y, test_size=0.3)
X_train.shape

def fit(X_train, Y_train):
  num=(X_train*Y_train).mean() -X_train.mean()*Y_train.mean()
  den=(X_train**2).mean() -X_train.mean()**2
  m=num/den
  c=Y_train.mean()-m*X_train.mean()
  return m,c

def predict(x,m,c):
  return m*x+c
def score(y_truth,y_pred):
  u=((y_truth-y_pred)**2).sum()
  v=((y_truth-y_truth.mean())**2).sum()
  return 1-u/v

def cost(x,y,m,c):
  return ((y - m*x - c)**2).mean()

m,c=fit(X_train,Y_train)
#test data
Y_test_pred=predict(X_test,m,c)
print("Test Score: ",score(Y_test,Y_test_pred))

#train data
Y_train_pred=predict(X_train,m,c)
print("Train Score: ",score(Y_train,Y_train_pred))
print("M , C ", m, c)

print("Cost on training data ", cost(X_train,Y_train,m,c))

