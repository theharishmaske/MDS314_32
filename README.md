# MDS314_32
Neural Networks 



------------------------------------------------------------------------------------------------------------------------------------------------
import numpy and panda library

```python
import numpy as np
import pandas as pd
```
---------------------------------------------------------------------------------------------------------------------------------------------------
all weightsand data are in the form

```python
df = pd.DataFrame([[(1,-1),1],[(1,1),1],[(-1,-1),-1],[(-1,1),1]])
# row,col = df.shape
weight=np.array([1,1])
wei=np.array([0,0])
```
-------------------------------------------------------------------------------------------------------------------------------------------------
McCulloch-pitts algorithm implementation python code

```python
def fun(x):
  if(x>=0):
   return 1
  else:
    return 0

def mpitt(weight,df):
  c_count=0
  w_count=0
  for i in range(len(df)):
    X=np.array(df.iloc[i,0])
    y_in=np.dot(X,weight.T)
    t_head=fun(y_in)
    target= df.iloc[i,1]
    if(target==t_head):
      c_count+=1
    else:
      w_count+=1
  print("Data point which are corrrectly classify are",c_count,"out of",len(df))
```


  -----------------------------------------------------------------------------------------------------------------------------------------------

Hebb Rule algorithm implementation

```pyhton
def fun(x):
  if(x>=0):
   return 1
  else:
    return 0
#hebb rule
def hebb(weight,df):
  row = len(df)
  b=0
  for i in range(row):
    X=np.array(df.iloc[i,0])
    y_in = np.dot(X,weight.T)+b
    t_head=fun(y_in)
    target=df.iloc[i,1]
    weight = weight + (X*target)
    b = b+target
  return [b,weight]
```

------------------------------------------------------------------------------------------------------------------------------------------------

Perceptron Algorithm Implementation

```python
def fun1(x,theta):
  if(x>theta):
    return 1
  elif(x<theta and x>(-1*theta)):
    return 0
  else:
    return -1
def perceptron(weight,df,alpha,epoch):
  epoch_2=0
  b=1
  while(epoch_2<=epoch):
    for i in range(len(df)):
      X=np.array(df.iloc[i,0])
      y_in = np.dot(X,weight.T)+b
      t_head=fun1(y_in,0)
      target=df.iloc[i,1]
      if(target==t_head):
        weight=weight
        b=b
      else:
        weight = weight + (alpha*target*X)
        b=b+(alpha*target)
    return [b,weight]
```

-----------------------------------------------------------------------------------------------------------------------------------------

Adaline algorithm implementation

```python
def adaline(weight,df,alpha,epoch):
  epoch_1 = 0
  b=1
  row=len(df)
  while (epoch_1<epoch):
    for i in range(row):
      X=np.array(df.iloc[i,0])
      target =df.iloc[i,1]
      y_in = np.dot(X, weight.T) + b
      t_head=y_in
      error = target-y_in
      if(np.all(error == 0)): # Use np.all() to check if all elements in error are zero
        print(b,weight)
        print(y_in)
      else:
        b=b+(alpha*error)
        weight=weight+(alpha*error*X)
        print(b,weight)
        print(y_in)
      epoch_1 +=1
    return [b,weight]
```
