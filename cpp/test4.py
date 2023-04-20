import numpy as np
k = [45]
k = np.array(k)

l = [1,2,3,4,5,6,76]
print(l[-1])
l = np.array(l)
print(all(item in k for item in l))