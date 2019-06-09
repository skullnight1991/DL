import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('data_linear.csv').values
x = np.array(data[:, 0]).reshape([-1,1])
y = np.array(data[:, 1]).reshape([-1,1])

def forward(x):
	return x * w

def loss(x, y):
	y_pred = forward(x)
	return (y_pred - y) * (y_pred - y)

l_func = np.zeros((20))
w_list = []
w = 1
for i in range(20):
    # r = y^ - y
    r = np.dot(x, w) - y
    print(np.sum(r*r))
    # Calculate loss function
    l_func[i] = 0.5 * np.sum(r*r) / len(x)
    
    # Update the weight
    w -= 0.0000001 * np.sum(np.multiply(x, 2) * (np.sum(r)))

    w_list.append(w)
		
print('%s %f' % ('Min weight=', min(w_list)))
print('%s %f' % ('Min loss=',l_func.min()))
#print(l_list)

minIndex = np.argmin(x)
maxIndex = np.argmax(x)

pred = np.dot(x, w)
plt.scatter(x,y)
plt.plot([x[minIndex], x[maxIndex]], [pred[minIndex], pred[maxIndex]] , 'r') 
plt.show()