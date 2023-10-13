import numpy as np
import random
from model import *
from func import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_num = 1
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
DOF = 4
cam_dist = 1
nf_size = 0.4
near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
tr = 0.8
chunksize = 2 ** 20  # Modify as needed to fit in GPU memory
select_data_amount = 10000
##########################################################################
# 0: rectuglar
# 1: ball-head
# 2: break ball-head
robot_id = 1

LOG_PATH ="sim_robo_%d/"%robot_id

test_model_name ='../train_log/final_model/sim_train_id1/best_model/best_model.pt'
os.makedirs(LOG_PATH+'/image',exist_ok=True)
os.makedirs(LOG_PATH+'/points',exist_ok=True)



data = np.load('../data/data_uniform_robo%d/1009(1)_con_dof4_data.npz'%robot_id)
num_raw_data = len(data["angles"])


sample_id = random.sample(range(num_raw_data), select_data_amount)
focal = torch.from_numpy(data['focal'].astype('float32'))

training_img = data['images'][sample_id[:int(select_data_amount * tr)]]
training_angles = data['angles'][sample_id[:int(select_data_amount * tr)]]

testing_img = data['images'][sample_id[int(select_data_amount * tr):]]
testing_angles = data['angles'][sample_id[int(select_data_amount * tr):]]

valid_amount = len(testing_angles)
print("test_data_sample_num:", valid_amount)

height, width = testing_img.shape[1:3]
rays_o, rays_d = get_rays(height, width, focal)

model = FBV_SM(d_input=5,
               # n_layers=4,
               d_filter=128,
               skip=(1,2),
               output_size=2)
model.load_state_dict(torch.load(test_model_name, map_location=torch.device(device)))

model.to(device)
model.eval()



def OM_eval(testing_img,testing_angles):
    threshold=0.4
    test_loss_list= []
    testing_img = torch.from_numpy(testing_img.astype('float32'))
    testing_angles = torch.from_numpy(testing_angles.astype('float32'))

    for v_i in range(valid_amount):
        angle = testing_angles[v_i]
        img_label = testing_img[v_i]

        outputs = model_forward(rays_o, rays_d,
                                near, far, model,
                                chunksize=chunksize,
                                arm_angle=angle,
                                DOF=DOF)

        rgb_predicted = outputs['rgb_map']
        rgb_each_point = outputs['rgb_each_point']
        rgb_each_point = outputs["rgb_each_point"].reshape(-1)
        all_points = outputs["query_points"].detach().cpu().numpy().reshape(-1, 3)
        rgb_each_point = rgb_each_point.where(rgb_each_point>0.1,torch.tensor(0).to(device))
        nonempty_idx = torch.nonzero(rgb_each_point).cpu().detach().numpy().reshape(-1)
        query_xyz = all_points[nonempty_idx]
        pose_matrix = pts_trans_matrix_numpy(angle[0],angle[1],no_inverse=False)
        query_xyz = np.concatenate((query_xyz, np.ones((len(query_xyz), 1))), 1)
        query_xyz = np.dot(pose_matrix, query_xyz.T).T[:, :3]

        img_label_tensor = img_label.reshape(-1).to(device)

        # rgb_predicted = (rgb_predicted > threshold).float()

        v_loss = torch.nn.functional.mse_loss(rgb_predicted, img_label_tensor)
        test_loss_list.append(v_loss.item())

        np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()

        np_image_combine = np.dstack((np_image, np_image, np_image))
        print(v_i,'loss:',v_loss.item())
        np_image_combine = np.clip(np_image_combine,0,1)
        if v_i < 100:
            plt.imsave(LOG_PATH + 'image/' + 'test%d.png'%v_i, np_image_combine)
            np.savetxt(LOG_PATH+'points/' + 'pts%d.csv'%v_i, query_xyz)

    np.savetxt(LOG_PATH+'test_loss_OM.csv',test_loss_list)

OM_eval(testing_img,testing_angles)

def NN_eval():
    test_loss_list = []
    os.makedirs(LOG_PATH + 'image_nn/',exist_ok= True)
    # for v_i in range(valid_amount):
    for v_i in range(100):

        angle = testing_angles[v_i]
        img_label = testing_img[v_i]
        distances = np.linalg.norm(training_angles - angle, axis=1)

        # Get the index of the nearest neighbor
        nearest_index = np.argmin(distances)

        predict_img = training_img[nearest_index]
        v_loss = np.mean((predict_img - img_label) ** 2)
        test_loss_list.append(v_loss)
        np_image_combine = np.dstack((predict_img, predict_img, predict_img))
        print(v_i,'loss:',v_loss)
        np_image_combine = np.clip( np_image_combine,0,1)

        if v_i < 100:
            plt.imsave(LOG_PATH + 'image_nn/' + 'test%d.png'%v_i, np_image_combine)

    np.savetxt(LOG_PATH+'test_loss_NN.csv', test_loss_list)

# NN_eval()

list_OM = np.loadtxt('sim_robo_1/test_loss.csv')
list_NN = np.loadtxt('sim_robo_1/test_loss_NN.csv')

print(np.mean(list_NN))
print(np.mean(list_OM))