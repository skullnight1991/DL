import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('Advertising.csv').values
x = np.array(data[:, 2])
y = np.array(data[:, 4])

# Make two list to stores lost values and weight values
l_list = []
w_list = []
w = 0
for i in range(200):
    # r = y^ - y
    r = (x * w) - y
    # Calculate loss function
    l_func = 0.5 * np.sum(r * r) / len(x)
    #print(l_func)
    
    # Update the weight
    w -= 0.000000001 * 2 * np.sum(x) * (np.sum(r))

    l_list.append(l_func)
    w_list.append(w)
		
print('%s %f' % ('Min weight=', min(w_list)))
print('%s %f' % ('Min loss=',l_func.min()))
print(w_list)
#print(l_list)

# Find the min and max index of data values
minIndex = np.argmin(x)
maxIndex = np.argmax(x)

# Plot
pred = np.dot(x, w)
plt.plot(w_list, l_list)
#plt.scatter(x,y)
#plt.plot([x[minIndex], x[maxIndex]], [pred[minIndex], pred[maxIndex]] , 'r') 
plt.show()
