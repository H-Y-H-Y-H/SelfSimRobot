import numpy as np

arr_list = []
for i in range(0,21,2):
    print(i)
    arr = np.loadtxt('new_action%d.csv'%i)
    print(len(arr))
    arr_list.append(arr)

arr= np.concatenate(arr_list)
print(len(arr))

np.savetxt('new_a.csv',arr)
