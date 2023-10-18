import numpy as np


def quan_eval():
    eval_folder = '%s_robo_%d(ee)/'%(sim_real,robot_id)
    list_OM = np.loadtxt(eval_folder+'test_loss_OM.csv')
    list_NN = np.loadtxt(eval_folder+'test_loss_NN.csv')
    list_RD = np.loadtxt(eval_folder+'test_loss_RS.csv')
    lists_4_eval = [list_RD,list_NN,list_OM]

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
    np.savetxt('paper_data/%s_paper_data%d(ee).csv'%(sim_real,robot_id),data_all)

import matplotlib.pyplot as plt

def fig_array():

    if arm_ee == 'ee':
        eval_folder = '%s_robo_%d(ee)/'%(sim_real,robot_id)
    else:
        eval_folder = '%s_robo_%d/' % (sim_real, robot_id)
    rs_img_pth = eval_folder+'/image_rs/'
    nn_img_pth = eval_folder+'/image_nn/'
    om_img_pth = eval_folder+'/image_om/'
    gt_img_pth = eval_folder+'/image_gt/'

    image_list = []
    h_gap = 5
    width = 100*4+3*h_gap
    v_gap = 10
    for i in range(2):
        img0 = plt.imread(rs_img_pth+'test%d.png'%i)
        img1 = plt.imread(nn_img_pth+'test%d.png'%i)
        img2 = plt.imread(om_img_pth+'test%d.png'%i)
        img3 = plt.imread(gt_img_pth+'test%d.png'%i)
        paddings = np.zeros((100,h_gap,4))
        img_line = np.hstack((img0,paddings,img1,paddings,img2,paddings,img3))

        image_list.append(img_line)
    paddings = np.zeros((v_gap,width,4))
    com_img = np.vstack((image_list[0],paddings,image_list[1]))
    plt.imsave('paper_data/quan%s_%d(%s).png'%(sim_real,robot_id,arm_ee),com_img)


robot_id = 1
sim_real = 'real'
arm_ee = 'ee'

# quan_eval()
# fig_array()

# Abnoraml Detection
# To fine-tune the model we use four different dataset
data_amount = [10,100,1000,10000]
def filter_data(list):
    new_list = []
    current_number = 100
    for i in list:
        if i<current_number:
            new_list.append(i)
            current_number=i
        else:
            new_list.append(current_number)
    while len(new_list)<50:
        new_list.insert(-1,new_list[-1])
    new_list = new_list[:50]
    return new_list



all_data = []
for i in range(len(data_amount)):
    list_curves = []
    for seed_n in range(3):
        y_data = np.loadtxt('../train_log/real_id2_%d(%d)_PE(arm)/log_val.txt'%(data_amount[i],seed_n))
        y_data = filter_data(y_data)
        list_curves.append(y_data)
    data_mean = np.mean(list_curves,axis=0)
    data_std = np.std(list_curves,axis=0)
    all_data.append(data_mean)
    all_data.append(data_std)


# np.savetxt('paper_data/resilience.csv', np.asarray(all_data).T)


import numpy as np
import matplotlib.pyplot as plt
color = ['#52c8fa','#bcf5ce','#ff9d00','#f55442']
plt.figure(figsize=(10, 6))
for i in range(4):
    # Sample data
    y = all_data[i*2]
    yerr = all_data[i * 2+1]

    x = list(range(len(y)))

    # Plot
    plt.plot(x, y, label='%d'%data_amount[i],color=color[i])
    plt.fill_between(x, y - yerr, y + yerr, color=color[i], alpha=0.4)

plt.xlabel('Fine-tuning Epochs',fontsize=14)
plt.ylabel('Prediction Error',fontsize=14)
# Increase the size of tick labels for both axes
plt.tick_params(axis='both', which='major', labelsize=14)

# Increase the size of the legend
plt.legend(fontsize=14)



plt.grid(True)
plt.tight_layout()
plt.show()


