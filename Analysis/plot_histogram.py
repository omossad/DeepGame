import numpy as np
arr = np.load('consecutives.txt.npy')
print(arr)
hist = np.zeros((10))
print(hist)
for i in arr:
    if i > 9 : 
        hist[9] = hist[9] + 1
    else:
        hist[i-1] = hist[i-1] + 1
print(hist)