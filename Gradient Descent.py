

import numpy as np

data =np.loadtxt("LR data.csv", delimiter=',')
data.shape

#M no.of training points
def step_gradient(points, learning_rate, m, c):
  m_slope=0
  c_slope=0
  #points=100 rows
  M=len(points)
  for i in range(M):
    #ith row 0th entry
    x=points[i,0]
    #ith row 1st entry
    y=points[i,1]
    m_slope += (-2/M)* (y-m* x -c)*x
    c_slope += (-2/M)* (y-m* x -c)
  new_m = m-learning_rate*m_slope
  new_c=c -learning_rate*c_slope
  return new_m,new_c

def gd(points, learning_rate, num_iterations):
  m=0
  c=0
  for i in range(num_iterations):
    m,c = step_gradient(points, learning_rate , m ,c)
    print(i," Cost: ",cost(points,m,c))
  return m,c

def cost(points, m, c):
  total_cost=0
  M=len(points)
  for i in range(M):
    x=points[i,0]
    y=points[i,1]
    total_cost += (1/M)*((y - m*x -c)**2)
  return total_cost

def run():
  data =np.loadtxt("LR data.csv", delimiter=',')
  learning_rate =0.0001
  num_iterations =100
  m,c =gd(data , learning_rate, num_iterations)
  print(m,c)

run()