import numpy as np
from env4 import FBVSM_Env
import pybullet as p
from train import *
from test_model import *
# changed Transparency in urdf, line181, Mar31

def scale_points(predict_points):
    scaled_points = predict_points / 4.  # scale /5  0.8
    scaled_points[:, [1, 2]] = scaled_points[:, [2, 1]]
    scaled_points[:, [1, 0]] = scaled_points[:, [0, 1]]
    scaled_points[:, 0] = -scaled_points[:, 0]
    scaled_points[:, 2] += 1.106

    new_points = scaled_points
    return new_points


# def load_offline_point_cloud(angle_list: list, debug_points, logger):
#     angle_lists = np.asarray([angle_list] * len(logger)) * 90  # -90 to 90
#     diff = np.sum(abs(logger - angle_lists), axis=1)
#     idx = np.argmin(diff)
#     predict_points = np.load(data_pth + '%04d.npy' % idx)
#     # scaled points
#
#     trans_points = scale_points(predict_points)
#     # test_points = np.random.rand(100, 3)
#     p_rgb = np.ones_like(predict_points)
#     p.removeUserDebugItem(debug_points)  # update points every step
#     debug_points = p.addUserDebugPoints(predict_points, p_rgb, pointSize=2)
#
#     return debug_points
#



def interact_env():
    DOF = 4
    test_name = 'log_160000data_in6_out1_img100(1)'
    test_model_pth = 'train_log/%s/best_model/'%test_name

    model, optimizer = init_models(d_input=(DOF-2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                   n_layers=4,
                                   d_filter=128,
                                   skip=(1,2 ),
                                   output_size=2)
    model.load_state_dict(torch.load(test_model_pth + "nerf.pt", map_location=torch.device(device)))
    model = model.to(torch.float64)
    model.eval()

    # start simulation:
    p.connect(p.GUI)

    env = FBVSM_Env(
        show_moving_cam=False,
        width=width,
        height=height,
        render_flag=True,
        num_motor=DOF)

    obs = env.reset()
    c_angle = obs[0]
    debug_points = 0
    action_space = 90

    # input para
    motor_input = []
    for m in range(DOF):
        motor_input.append(p.addUserDebugParameter("motor%d:"%m, -1, 1, 0))

    for i in range(100000):
        for dof_i in range(DOF):
            c_angle[dof_i] = p.readUserDebugParameter(motor_input[dof_i])
        degree_angles = c_angle*action_space
        occu_pts = test_model(degree_angles,model)
        p_rgb = np.ones_like(occu_pts)
        p.removeUserDebugItem(debug_points)  # update points every step
        debug_points = p.addUserDebugPoints(occu_pts, p_rgb, pointSize=2)
        obs, _, _, _ = env.step(c_angle)



#
# def interact_env_offline(
#         pic_size: int = 100,
#         render: bool = True,
#         interact: bool = True,
#         dof: int = 2,
#         runTimes = 10000):  # 4dof
#
#     PATH = 'train_log/log_400data_in6_out1_img100(1)/visual_test01/'
#     data_pth = PATH + 'pc_record/'
#     # data_pth = PATH + 'visual/pc_record/'
#
#
#     p.connect(p.GUI) if render else p.connect(p.DIRECT)
#     logger = np.loadtxt(PATH + "logger.csv")
#     env = FBVSM_Env(
#         show_moving_cam=False,
#         width=pic_size,
#         height=pic_size,
#         render_flag=render,
#         num_motor=dof)
#
#     obs = env.reset()
#     c_angle = obs[0]
#     debug_points = 0
#
#     if interact:
#         # input para
#         motor_input = []
#         for m in range(DOF):
#             motor_input.append(p.addUserDebugParameter("motor%d:" % m, -1, 1, 0))
#
#         for i in range(runTimes):
#             for dof_i in range(dof):
#                 c_angle[dof_i] = p.readUserDebugParameter(motor_input[dof_i])
#             debug_points = load_offline_point_cloud(c_angle,
#                                             debug_points,
#                                             logger)
#
#             obs, _, _, _ = env.step(c_angle)
#             # print(obs[0])


if __name__ == "__main__":



    interact_env()
