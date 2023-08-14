import numpy as np
from env4 import FBVSM_Env
import pybullet as p
from train import *
from test_model import *
# changed Transparency in urdf, line181, Mar31

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



if __name__ == "__main__":
    interact_env()
