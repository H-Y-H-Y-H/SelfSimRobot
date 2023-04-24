import numpy as np
import matplotlib.pyplot as plt

PATH = "train_log/log_1600data_in5_out1_img100(3)/"   # 2 dof
# PATH = "train_log/log_64000data_in6_out1_img100(3)/"   # 3 dof

train_data = []
with open(PATH + "log_train.txt") as f:
    lines = f.readlines()

for line in lines:
    number = line.split("(")[1].split(",")[0]
    train_data.append(float(number))


valid_data = np.loadtxt(PATH+"log_val.txt")

plt.figure()
plt.plot(train_data, label="train")
plt.plot(valid_data, label="validation")
plt.legend()
plt.title("2-dof train and validation (-log10 loss)")
plt.show()

