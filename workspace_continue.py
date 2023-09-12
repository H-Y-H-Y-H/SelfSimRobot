import numpy as np

action_list = []

j_flag,k_flag,l_flag = -1,-1,-1

action_true_list = np.loadtxt('workspace_robo0_dof4_size20.csv')
action_true_list *=10
action_true_list = np.asarray(action_true_list,dtype=int)

# -1, -0.9, .. 0, 0.1, 0.2, ..., 0.9, 1
for i in range(21):
    cmd_0 = -10 + (i)

    j_flag *= -1
    for j in range(21):
        cmd_1 = -10 + j
        cmd_1 *= j_flag

        k_flag *= -1
        for k in range(21):
            cmd_2 = -10 + k
            cmd_2*=k_flag

            l_flag*= -1
            for l in range(21):
                cmd_3 = -10 + l
                cmd_3 *= l_flag

                act_cmd = [cmd_0,cmd_1,cmd_2,cmd_3]
                act_cmd_arr =np.asarray([act_cmd]*len(action_true_list))
                act_cmd_arr *= 10
                act_cmd_arr = np.asarray(act_cmd_arr, dtype=int)

                if ((act_cmd_arr - action_true_list).sum(1) == 0).any():
                    print(1)
                    action_list.append(act_cmd)


                print(act_cmd)


np.savetxt('new_action.csv',action_list)


