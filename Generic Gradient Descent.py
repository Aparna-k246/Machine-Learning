
from sklearn.datasets import load_boston
import numpy as np
data=load_boston()


def step_gradient(X, Y, learning_rate, m):
    m_slope=np.array([0 for i in range(len(X[0]))])
    M=len(X)
    for i in range(M):
        x=X[i]
        y=Y[i]
        for j in range(len(m_slope)):
            m_slope[j]+=(-2/M)*(y-((m*x).sum()))*x[j]
    new_m=m-learning_rate*m_slope
    cost(new_m, X, Y)
    return new_m
    
def cost(m, X, Y):
    cst=0
    for i in range(len(X)):
        cst+=(2/len(X))*(Y[i]-sum(m*X[i]))
    print(cst)
    
    
def gd(X, Y, iterations, learning_rate):
    m=np.array([0 for i in range(len(X[0]))])
    for i in range(iterations):
        m=step_gradient(X, Y, learning_rate, m)
    return m



def run():
    X=data.data
    Y=data.target
    X=np.append(X, np.ones(len(X), dtype='int').reshape(-1, 1), axis=1)
    iterations=500
    learning_rate=0.000001
    m=gd(X, Y, iterations, learning_rate)

run()