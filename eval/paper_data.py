import numpy as np

robot_id = 0

list_OM = np.loadtxt('sim_robo_%d/test_loss_OM.csv'%robot_id)
list_NN = np.loadtxt('sim_robo_%d/test_loss_NN.csv'%robot_id)
list_RD = np.loadtxt('sim_robo_%d/test_loss_RD.csv'%robot_id)
lists_4_eval = [list_OM,list_NN,list_RD]

mean_list = []
std_list = []
max_list = []
min_list = []

for i in range(len(lists_4_eval)):
    mean_list.append(np.mean(lists_4_eval[i]))
    std_list.append(np.std(lists_4_eval[i]))
    max_list.append(np.max(lists_4_eval[i]))
    min_list.append(np.min(lists_4_eval[i]))

data_all = np.concatenate((mean_list, std_list, max_list, min_list)).reshape(4,3).T
np.savetxt('paper_data%d.csv'%robot_id,data_all)

