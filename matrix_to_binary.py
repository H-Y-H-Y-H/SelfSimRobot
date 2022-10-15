import numpy as np

data = np.random.random(64)
for i in range(len(data)):
    if data[i] > 0.5:
        data[i] = 0
    else:
        data[i] = 1
print(data)
