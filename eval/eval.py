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
test_num = 2000
##########################################################################
# 0: rectuglar
# 1: ball-head
# 2: break ball-head
robot_id = 1
sim_real = 'real'
arm_ee = 'ee'
LOG_PATH ="%s_robo_%d(%s)/"%(sim_real,robot_id,arm_ee)

test_model_name ='../train_log/%s_id%d_10000(1)_PE(%s)/best_model/best_model.pt'%(sim_real,robot_id,arm_ee)
os.makedirs(LOG_PATH+'/image_om',exist_ok=True)

data = np.load('../data/%s_data/%s_data_robo%d(%s).npz'%(sim_real,sim_real,robot_id,arm_ee))
num_raw_data = len(data["angles"])


sample_id = random.sample(range(num_raw_data), select_data_amount)
# Create a list of all possible indices
all_indices = list(range(num_raw_data))

# Calculate the complement of A (indices not in A)
test_sample_id = list(set(all_indices) - set(sample_id))
test_sample_id = random.sample(test_sample_id, test_num)


focal = torch.from_numpy(data['focal'].astype('float32'))

training_img = data['images'][sample_id[:int(select_data_amount * tr)]]
training_angles = data['angles'][sample_id[:int(select_data_amount * tr)]]

# testing_img = data['images'][sample_id[int(select_data_amount * tr):]]
# testing_angles = data['angles'][sample_id[int(select_data_amount * tr):]]
testing_img = data['images'][test_sample_id]
testing_angles = data['angles'][test_sample_id]

test_amount = len(testing_angles)
print("test_data_sample_num:", test_amount)

height, width = testing_img.shape[1:3]
rays_o, rays_d = get_rays(height, width, focal)

encoder = PositionalEncoder(d_input=5, n_freqs=10, log_space=True)

model = FBV_SM(encoder = encoder)

model.load_state_dict(torch.load(test_model_name, map_location=torch.device(device)))

model.to(device)
model.eval()



def OM_eval(testing_img,testing_angles):

    test_loss_list= []
    testing_img = torch.from_numpy(testing_img.astype('float32'))
    testing_angles = torch.from_numpy(testing_angles.astype('float32'))

    for v_i in range(test_amount):
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
            plt.imsave(LOG_PATH + 'image_om/' + 'test%d.png'%v_i, np_image_combine)
            # np.savetxt(LOG_PATH+'points/' + 'pts%d.csv'%v_i, query_xyz)

    np.savetxt(LOG_PATH+'test_loss_OM.csv',test_loss_list)

OM_eval(testing_img,testing_angles)

def save_gt():
    os.makedirs(LOG_PATH + 'image_gt/', exist_ok=True)
    angle_list = []
    for v_i in range(test_amount):
        angle = testing_angles[v_i]
        img_label = testing_img[v_i]
        img_label = np.dstack((img_label, img_label, img_label))
        img_label = np.clip( img_label,0,1)
        if v_i ==100:
            break
        angle_list.append(angle)
        plt.imsave(LOG_PATH + 'image_gt/' + 'test%d.png' % v_i, img_label)
    np.savetxt(LOG_PATH+'test_angles.csv',angle_list)
save_gt()
def NN_RD_eval():
    test_loss_list = []
    rdm_loss_list = []

    os.makedirs(LOG_PATH + 'image_nn/',exist_ok= True)
    os.makedirs(LOG_PATH + 'image_rs/',exist_ok= True)

    # for v_i in range(valid_amount):
    for v_i in range(test_amount):

        angle = testing_angles[v_i]
        img_label = testing_img[v_i]
        distances = np.linalg.norm(training_angles - angle, axis=1)

        # Get the index of the nearest neighbor
        nearest_index = np.argmin(distances)
        rdm_idx = random.randint(0, len(training_img))
        predict_img = training_img[nearest_index]
        rdm_img = training_img[rdm_idx]

        v_loss = np.mean((predict_img - img_label) ** 2)
        test_loss_list.append(v_loss)
        rd_loss = np.mean((rdm_img - img_label) ** 2)
        rdm_loss_list.append(rd_loss)

        print(v_i,'loss NN & RS:',v_loss, rd_loss)


        np_image_combine = np.dstack((predict_img, predict_img, predict_img))
        np_image_combine = np.clip( np_image_combine,0,1)

        rdm_img= np.dstack((rdm_img, rdm_img, rdm_img))
        rdm_img = np.clip( rdm_img,0,1)

        if v_i < 100:
            plt.imsave(LOG_PATH + 'image_nn/' + 'test%d.png'%v_i, np_image_combine)
            plt.imsave(LOG_PATH + 'image_rs/' + 'test%d.png'%v_i, rdm_img)

    np.savetxt(LOG_PATH+'test_loss_NN.csv', test_loss_list)
    np.savetxt(LOG_PATH+'test_loss_RS.csv', rdm_loss_list)


NN_RD_eval()



