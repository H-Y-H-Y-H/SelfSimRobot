import time
import pyautogui
import numpy as np
import torch.optim as optim
from env4 import FBVSM_Env
import pybullet as p
from train import *
from test_model import *
import random

# changed Transparency in urdf, line181, Mar31

def interact_env(mode=0,n_samples=10):

    c_angle = env.init_obs[0]
    debug_points = 0
    t = np.linspace(0, 1, n_samples).reshape(-1, 1)


    if mode == 1:
        # traj_list = np.loadtxt('planning/trajectory/spiral.csv')
        # color_list = np.ones_like(traj_list)
        # p.addUserDebugPoints(traj_list, color_list, pointSize=3)
        #
        #
        # cmds = np.loadtxt('planning/trajectory/cmds_output.csv')
        # for i in range(0,len(cmds),10):
        #     print(cmds[i])
        #     obs, _, _, _ = env.step(cmds[i])


        for i in range(1,len(cmds)):
            print(cmds[i])
            traj_array = (1 - t) * cmds[i-1] + t * cmds[i]
            for traj in traj_array:
                obs, _, _, _ = env.step(traj)


    # input para
    motor_input = []
    for m in range(DOF):
        motor_input.append(p.addUserDebugParameter("motor%d:"%m, -1, 1, env.get_obs()[0][m]))


    c_angle = torch.tensor(c_angle).to(device)
    debug_points_buffer = []
    angles_logger = []
    sep = 100
    for i in range(500):
        # for dof_i in range(DOF):
        #     c_angle[dof_i] = p.readUserDebugParameter(motor_input[dof_i])

        c_angle = angles_input[i//sep] + (angles_input[i//sep+1] - angles_input[i//sep])*(i%sep)/sep
        angles_logger.append(c_angle)
        c_angle = torch.tensor(c_angle).to(device)
        degree_angles = c_angle*action_space

        # occu_pts = test_model(degree_angles,model)
        # t1 = time.time()
        occu_pts = query_models(degree_angles,model,DOF)
        # t2 = time.time()
        # print("FPS:",1/(t2-t1))
        occu_pts = occu_pts.detach().cpu().numpy()

        if len(occu_pts)>10000:
            idx = np.arange(len(occu_pts))
            np.random.shuffle(idx)
            occu_pts = occu_pts[idx[:10000]]

        p_rgb = [(0,1,0.5)]*len(occu_pts)
        p_rgb = np.asarray(p_rgb)

        p.removeUserDebugItem(debug_points)  # update points every step
        debug_points = p.addUserDebugPoints(occu_pts, p_rgb, pointSize=8)

        # debug_points = p.addUserDebugPoints([occu_pts], [p_rgb], pointSize=10)

        c_angle_cmd = c_angle.detach().cpu().numpy()
        obs, _, _, _ = env.step(c_angle_cmd)

        #Saving:
        screenshot = pyautogui.screenshot()
        # Save the screenshot
        screenshot.save("data/paper_fig_eval3d/%d.png"%i)
        np.savetxt('data/paper_fig_eval3d/robot0.csv',angles_logger)


def go_to_target_pos():
    obs = env.reset()
    c_angle = obs[0]

    # Loss threshold
    threshold = 1e-5
    max_iterations = 1000
    pred_occ_point_visual = 0
    loss_fc = nn.MSELoss()

    traj_list = np.loadtxt('planning/trajectory/spiral.csv')
    color_list = np.ones_like(traj_list)
    p.addUserDebugPoints(traj_list, color_list, pointSize=3)

    # based on the ee compute the joint commands
    # action_array = np.load('data/real_data/real_data0920_robo1_166855(ee).npz')['angles']

    cmd_tensor = torch.tensor(c_angle, requires_grad=True)
    # Define the optimizer
    optimizer = optim.Adam([cmd_tensor], lr=0.04)
    cmds_output_list = []
    for a_n in range(len(traj_list)):
        target_pos = traj_list[a_n] #[0.1, 0.2, 0.18] #

        show_target_point = p.addUserDebugPoints([target_pos], [[1,0,0]], pointSize=10)

        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float64, requires_grad=False).to(device)

        loss = 0
        for j in range(max_iterations):
            optimizer.zero_grad()  # Clear previous gradients

            degree_angles = cmd_tensor * action_space
            occu_pt_pred = query_models(degree_angles,model,DOF,mean_ee=True)

            p.removeUserDebugItem(pred_occ_point_visual)  # update points every step
            occu_pt_pred_arr = occu_pt_pred.detach().cpu().numpy()
            pred_occ_point_visual = p.addUserDebugPoints([occu_pt_pred_arr], [[0,1,0]], pointSize=10)

            # Compute the loss as the Euclidean distance between the target and the current position
            loss = loss_fc(occu_pt_pred, target_pos_tensor)
            if loss.item() == np.nan:
                break
            print('loss:',loss.item())
            if loss.item() < threshold:
                print(f"Converged at iteration {j}")
                break

            # Compute the gradients and perform an optimization step
            loss.backward()
            optimizer.step()
            # optimized_cmd = cmd_tensor.detach().numpy()
            # obs, _, _, _ = env.step(optimized_cmd)

        if loss.item() == np.nan:
            continue
        # The optimized joint angles
        optimized_cmd = cmd_tensor.detach().numpy()
        obs, _, _, _ = env.step(optimized_cmd)
        cmds_output_list.append(np.copy(optimized_cmd))
        np.savetxt('planning/trajectory/cmds_output.csv',cmds_output_list)

def points_in_obstacle(points,
                       box_origin=torch.tensor([0.25, 0, -0.1], dtype=torch.float32).to(device),
                       box_dims=torch.tensor([0.3, 0.02, 0.5], dtype=torch.float32).to(device)):

    half_dims = box_dims / 2.0

    # Define the lower and upper bounds of the box.
    lower_bounds = box_origin - half_dims
    upper_bounds = box_origin + half_dims

    # Adjust lower_bounds and upper_bounds for the height (z) as the origin is at the center of the bottom
    lower_bounds[2] = box_origin[2]
    upper_bounds[2] = box_origin[2] + box_dims[2]

    # Check if points are inside the box.
    mask = torch.all((points >= lower_bounds) & (points <= upper_bounds), dim=1)

    # Return the points inside the box.
    return points[mask].sum()


def collision_free_planning(n=5):

    # Create initial trajectory points (4 points in this case)
    start_angle = env.init_obs[0]
    target_angle = np.copy(start_angle)
    target_angle[0] = target_angle[0]*(-1)

    # Average them
    t = np.linspace(0, 1, n).reshape(-1, 1)
    print(t)
    traj_array = (1 - t) * start_angle + t * target_angle
    middle_tensor = torch.tensor(traj_array[1:-1], requires_grad=True)

    # Loss threshold
    num_epoch = 100
    sub_step = 10

    w_collision = 1.0  # Adjust based on your requirements
    w_smoothness = 1  # Adjust based on your requirements
    w_efficiency = 1  # Adjust based on your requirements

    # Define the optimizer
    optimizer = optim.SGD([middle_tensor], lr=0.02)
    step_id = torch.linspace(0, 1, sub_step).reshape(-1, 1)

    for epoch in range(num_epoch):

        total_loss = torch.tensor(0, dtype=torch.float64, requires_grad=True).to(device)
        collision_loss = torch.tensor(0, dtype=torch.float64, requires_grad=True).to(device)
        optimizer.zero_grad()  # Clear previous gradients
        list_pred_occ_point_visual = []

        for a_n in range(len(middle_tensor)+1):
            if a_n == 0:
                sub_strt_angle = traj_array[0]
            else:
                sub_strt_angle = middle_tensor[a_n-1]

            if a_n == len(middle_tensor):
                sub_trgt_angle = traj_array[-1]
            else:
                sub_trgt_angle = middle_tensor[a_n]  # [0.1, 0.2, 0.18]

            step_angle = (1 - step_id) * sub_strt_angle + step_id * sub_trgt_angle

            for i in range(len(step_angle)):
                angle_i = step_angle[i]
                angle_i *= action_space
                occu_pt_pred = query_models(angle_i, model, DOF, mean_ee=False)

                # points_in_obstacles return the number of points got collision.
                count = points_in_obstacle(points=occu_pt_pred)
                collision_loss += count

                occu_pt_pred_arr = occu_pt_pred.detach().cpu().numpy()

                if len(occu_pt_pred_arr) > 10000:
                    idx = np.arange(len(occu_pt_pred_arr))
                    np.random.shuffle(idx)
                    occu_pt_pred_arr = occu_pt_pred_arr[idx[:10000]]
                p_rgb = np.ones_like(occu_pt_pred_arr)
                list_pred_occ_point_visual.append(p.addUserDebugPoints(occu_pt_pred_arr, p_rgb, pointSize=2))

        # smoothness_loss = torch.sum(torch.pow(middle_tensor[1:] - middle_tensor[:-1], 2))
        straight_line = (1 - t) * start_angle + t * target_angle
        efficiency_loss = torch.sum(torch.pow(middle_tensor - torch.tensor(straight_line[1:-1]), 2)).to(device)
        smoothness_loss = torch.sum(torch.pow(middle_tensor[1:] - middle_tensor[:-1], 2))

        total_loss = w_collision * collision_loss + w_efficiency * efficiency_loss + w_smoothness * smoothness_loss

        total_loss.backward()
        optimizer.step()
        middle_tensor.data.clamp_(-0.5, 0.5)

        print(epoch,'total_loss',total_loss.item(), 'collision_loss', collision_loss.item(), 'efficiency_loss',
              efficiency_loss.item(), 'smoothness_loss', smoothness_loss.item())
        if collision_loss.item() == 0:
            break

        for k in range(len(list_pred_occ_point_visual)):
            p.removeUserDebugItem(list_pred_occ_point_visual[k])  # update points every step

        middle_points = middle_tensor.detach().cpu().numpy()
        traj_array[1:-1] = middle_points
        np.savetxt('planning/trajectory/free_collision_planning.csv',traj_array)

    middle_points = middle_tensor.detach().cpu().numpy()
    traj_array[1:-1] = middle_points
    np.savetxt('planning/trajectory/free_collision_planning.csv', traj_array)
    for s in range(len(traj_array)):
        obs, _, _, _ = env.step(traj_array[s])

        # optimized_cmd = cmd_tensor.detach().numpy()
        # obs, _, _, _ = env.step(optimized_cmd)

        # # The optimized joint angles
        # optimized_cmd = cmd_tensor.detach().numpy()
        # obs, _, _, _ = env.step(optimized_cmd)
        # cmds_output_list.append(np.copy(optimized_cmd))
        # np.savetxt('planning/trajectory/cmds_output.csv',cmds_output_list)



import heapq

class Node:
    def __init__(self, value, parent=None, g=0., h=0.):
        self.value = value
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def heuristic(angle_a, target_ee):
    angle_a = torch.tensor(angle_a, dtype=torch.float64, requires_grad=False).to(device)

    angle_a = angle_a * action_space
    a_ee_pt = query_models(angle_a, model_ee, DOF, mean_ee=True).detach().cpu().numpy()
    diff = a_ee_pt - target_ee
    # weighted_diff = diff * weights
    return np.linalg.norm(diff)


def heuristic_angle(angle_a, angle_b):
    angle_a = torch.tensor(angle_a, dtype=torch.float64, requires_grad=False).to(device)
    angle_b = torch.tensor(angle_b, dtype=torch.float64, requires_grad=False).to(device)
    angle_a = angle_a * action_space
    angle_b = angle_b * action_space

    a_ee_pt = query_models(angle_a, model_ee, DOF, mean_ee=True).detach().cpu().numpy()
    b_ee_pt = query_models(angle_b, model_ee, DOF, mean_ee=True).detach().cpu().numpy()

    diff = a_ee_pt - b_ee_pt
    # weighted_diff = diff * weights
    return np.linalg.norm(diff)


def get_neighbors(node_value):
    node_np = np.array(node_value)

    # Generate potential increments and decrements for each joint angle
    increments = np.eye(len(node_np)) * joint_interval
    decrements = -increments

    # Get potential neighbors by adding/subtracting 0.1 from the joint angles
    potential_neighbors = np.vstack([node_np + increments, node_np + decrements])

    # Filter to ensure all angles are within the range [-1, 1]
    valid_neighbors = potential_neighbors[
        np.all(np.logical_and(potential_neighbors >= -1.0, potential_neighbors <= 1.0), axis=1)]

    return [tuple(v) for v in valid_neighbors]


def is_collision(node_value,action_space =90):

    node_value_degree = (torch.tensor(node_value)* action_space).to(device)

    occu_pt_pred = query_models(node_value_degree, model, DOF, mean_ee=False)

    occu_pts = occu_pt_pred.detach().cpu().numpy()

    if len(occu_pts) > 10000:
        idx = np.arange(len(occu_pts))
        np.random.shuffle(idx)
        occu_pts = occu_pts[idx[:10000]]

    p_rgb = np.ones_like(occu_pts)
    debug_points_list.append(p.addUserDebugPoints(occu_pts, p_rgb, pointSize=2))


    # points_in_obstacles return the number of points got collision.
    count = points_in_obstacle(points=occu_pt_pred)
    # Dummy function; replace with your collision check

    if count.item() > 0:
        Collision_flag = True
    else:
        Collision_flag = False

    # print('Collision: ', Collision_flag)

    return Collision_flag


def A_star_search(start, goal):
    threshold = 0.1
    open_list = []
    closed_list = set()
    open_nodes_dict = {}

    start_a = torch.tensor(start, dtype=torch.float64, requires_grad=False).to(device)
    goal_a = torch.tensor(goal, dtype=torch.float64, requires_grad=False).to(device)
    start_a = start_a * action_space
    goal_a = goal_a * action_space
    start_ee = query_models(start_a, model_ee, DOF, mean_ee=True).detach().cpu().numpy()
    goal_ee = query_models(goal_a, model_ee, DOF, mean_ee=True).detach().cpu().numpy()
    p.addUserDebugPoints([start_ee], [[0, 1, 0]], pointSize=10)
    p.addUserDebugPoints([goal_ee], [[0, 1, 0]], pointSize=10)


    start_node = Node(start, g=0, h=heuristic(start, goal_ee))
    heapq.heappush(open_list, start_node)

    while open_list:
        print('node num: ',len(open_list))
        current_node = heapq.heappop(open_list)
        print(round(current_node.f,4), current_node.value, )

        if tuple(current_node.value) in closed_list:
            continue
        closed_list.add(tuple(current_node.value))

        # Check if reached goal
        if np.linalg.norm(np.array(current_node.value) - np.array(goal)) < threshold:
            path = []
            while current_node:
                path.insert(0, current_node.value)
                current_node = current_node.parent
            return path

        neighbors = get_neighbors(current_node.value)

        for neighbor in neighbors:
            if tuple(neighbor) in closed_list:
                # print('--Existed Nodes')
                continue
            elif is_collision(neighbor):
                # print('--Collision nodes')
                for rm_pts in debug_points_list:
                    p.removeUserDebugItem(rm_pts)
                continue

            g = heuristic(neighbor, start_ee)
            h = heuristic(neighbor, goal_ee)
            neighbor_node = Node(neighbor, current_node, g, h)

            # Check if the neighbor is already in the open_list with a higher f value
            if tuple(neighbor) in open_nodes_dict:
                if open_nodes_dict[tuple(neighbor)].f > neighbor_node.f:
                    # Remove the old node from the open_list
                    open_list.remove(open_nodes_dict[tuple(neighbor)])
                    heapq.heapify(open_list)  # Re-heapify after removal
                    # Add the new node with the better f value
                    heapq.heappush(open_list, neighbor_node)
                    open_nodes_dict[tuple(neighbor)] = neighbor_node
            else:
                heapq.heappush(open_list, neighbor_node)
                open_nodes_dict[tuple(neighbor)] = neighbor_node
    return None


class RRTNode:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent


def find_nearest_node(tree, point):
    # Convert tree values to a numpy array
    tree_values = np.array([node.value for node in tree])

    # Compute distances from every node to the point
    distances = np.linalg.norm(tree_values - np.array(point), axis=1)

    # Get the index of the nearest node
    nearest_idx = np.argmin(distances)

    return tree[nearest_idx]


def rrt(start, goal, max_iterations=1000):
    tree = [RRTNode(start)]

    for i in range(max_iterations):
        print(i)
        random_point = tuple(random.uniform(-1, 1) for _ in start)

        nearest_node = find_nearest_node(tree, random_point)

        # Calculate direction from nearest node to random point
        direction = np.array(random_point) - np.array(nearest_node.value)
        direction /= np.linalg.norm(direction)  # normalize

        # Extend the tree towards the random point by joint_interval
        new_point = tuple(np.array(nearest_node.value) + joint_interval * direction)

        if not is_collision(new_point):  # Ensure you have a collision checker function
            new_node = RRTNode(new_point, nearest_node)
            tree.append(new_node)

            if heuristic_angle(new_point, goal) < joint_interval:
                goal_node = RRTNode(goal, new_node)
                tree.append(goal_node)
                return tree  # Path found

    return None  # If after max_iterations, path is still not found

def reconstruct_path(rrt_tree, goal):
    path = []
    current_node = rrt_tree[-1]  # Assuming the last node in tree is the goal node
    while current_node:
        path.insert(0, current_node.value)
        current_node = current_node.parent
    return path



def is_collision_line(point1, point2, num_samples=10):
    for i in np.linspace(0, 1, num_samples):
        interpolated_point = tuple(a * (1 - i) + b * i for a, b in zip(point1, point2))
        if is_collision(interpolated_point):
            return True
    print('cut')
    return False


def shortcut_path(path):
    i = 0
    while i < len(path) - 2:
        print(i)
        pointA = path[i]
        skip = 2
        while (i + skip) < len(path):
            pointB = path[i + skip]
            if not is_collision_line(pointA, pointB):  # Make sure you implement this function
                del path[i+1:i+skip]
                break
            skip += 1
        i += 1
    return path




if __name__ == "__main__":
    DOF = 4
    seed = 1
    action_space = 90

    robot_id = 0
    sim_real = 'real'
    EndeffectorOnly = False
    both_models = False
    data_amount = 10000
    # test_name = 'real_train_1_log0928_%ddof_%d(%d)/' % (data_point, 100, seed)
    # test_model_pth = 'train_log/%s/best_model/' % test_name

    # test_model_pth = 'train_log/%s_id%d_%d(%d)_PE(arm)/best_model/'%(sim_real,robot_id,data_amount,seed)
    test_model_pth = 'train_log/%s_id%d_%d(%d)_PE(arm)/best_model/'%(sim_real,robot_id,data_amount,seed)
    # test_model_pth = 'train_log/%s_id1_10000(%d)_PE/best_model/'%(sim_real,seed)


    # DOF + 3 -> xyz and angle2 or 3 -> xyz
    model, optimizer = init_models(d_input=(DOF - 2) + 3,
                                   d_filter=128,
                                   output_size=2,
                                   FLAG_PositionalEncoder=True)
    model.load_state_dict(torch.load(test_model_pth + "best_model.pt", map_location=torch.device(device)))
    model = model.to(torch.float64)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


    test_model_ee_pth = 'train_log/%s_id%d_10000(%d)_PE(ee)/best_model/' % (sim_real, robot_id, seed)
    if EndeffectorOnly:
        model, _ = init_models(d_input=(DOF - 2) + 3,
                                  d_filter=128,
                                  output_size=2,
                                  FLAG_PositionalEncoder=True)
        model.load_state_dict(torch.load(test_model_ee_pth + "best_model.pt", map_location=torch.device(device)))
        model = model.to(torch.float64)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    if both_models:
        model_ee, _ = init_models(d_input=(DOF - 2) + 3,
                                  d_filter=128,
                                  output_size=2,
                                  FLAG_PositionalEncoder=True)
        model_ee.load_state_dict(torch.load(test_model_ee_pth + "best_model.pt", map_location=torch.device(device)))
        model_ee = model_ee.to(torch.float64)
        model_ee.eval()
        for param in model_ee.parameters():
            param.requires_grad = False

    # start simulation:
    p.connect(p.GUI)
    p.configureDebugVisualizer(rgbBackground=[1, 1, 1])
    camera_distance = 5
    camera_pitch = -30
    camera_yaw = 0
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                 cameraPitch=camera_pitch,
                                 cameraYaw=camera_yaw,
                                 cameraTargetPosition=[0, 0, 0])
    rotation_speed = 10  # Degrees per second
    start_time = time.time()

    MODE = 0

    if MODE == -1:
        path = list(np.loadtxt('planning/trajectory/fcl_169_smooth.csv'))
        path = shortcut_path(path)
        print('cutted_path:',path)
        # path = smooth_path_bspline(path)
        print('DONE:')
        np.savetxt('planning/trajectory/fcl_169_smooth.csv',path)
        print(path)


    elif MODE == 0:

        angles_input = np.loadtxt('eval/%s_robo_%d/test_angles.csv'%(sim_real,robot_id))/90
        # angles_input = np.loadtxt('train_log/real_id2_10000(1)_PE(arm)/image/valid_angle.csv')[4]/90
        env = FBVSM_Env(
            show_moving_cam=False,
            robot_ID=robot_id,
            width=width,
            height=height,
            render_flag=True,
            num_motor=DOF,
            dark_background=True,
            # init_angle=angles_input,
            rotation_view = False
        )


        # cmds = np.loadtxt('planning/trajectory/fcl_169.csv')
        interact_env(0)

    elif MODE == 1:
        env = FBVSM_Env(
            show_moving_cam=False,
            robot_ID=robot_id,
            width=width,
            height=height,
            render_flag=True,
            num_motor=DOF,
            dark_background=True,
            init_angle=[0, 1, -1, -1])

        go_to_target_pos()

    elif MODE == 2:
        env = FBVSM_Env(
            show_moving_cam=False,
            robot_ID=robot_id,
            width=width,
            height=height,
            render_flag=True,
            num_motor=DOF,
            dark_background=True,
            init_angle=[-0.5, -0.3, -0.5, -0.2])

        # -45.4737, -30.3158, -49.2632, -18.9474

        env.add_obstacles('planning/obstacles/urdf/obstacles.urdf',
                          [0.25, 0, -0.1],
                          p.getQuaternionFromEuler([0, 0, 0]))

        cmds = np.loadtxt('planning/trajectory/fcl_169_smooth.csv')

        interact_env(1)

        # collision_free_planning()

    # A star
    elif MODE == 3:
        # Example usage:
        start = (-0.5, -0.3, -0.5, -0.2)
        goal =  ( 0.5, -0.3, -0.5, -0.2)
        # weights = np.array([10.0, 0.8, 0.6, 0.4])

        env = FBVSM_Env(
            show_moving_cam=False,
            robot_ID=robot_id,
            width=width,
            height=height,
            render_flag=True,
            num_motor=DOF,
            dark_background=True,
            init_angle=start)
        env.add_obstacles('planning/obstacles/urdf/obstacles.urdf',
                          [0.25, 0, -0.1], p.getQuaternionFromEuler([0, 0, 0]))

        debug_points_list = []

        joint_interval = 0.2
        path = A_star_search(start, goal)

        t = np.linspace(0, 1, 10).reshape(-1, 1)
        print(path)
        np.savetxt('planning/trajectory/fcl_%d.csv'%int(time.time()),path)

    elif MODE == 4:
        # RRT
        debug_points_list = []
        start = (-0.5, -0.3, -0.5, -0.2)
        goal =  ( 0.5, -0.3, -0.5, -0.2)
        joint_interval = 0.1
        env = FBVSM_Env(
            show_moving_cam=False,
            robot_ID=robot_id,
            width=width,
            height=height,
            render_flag=True,
            num_motor=DOF,
            dark_background=True,
            init_angle=start)
        env.add_obstacles('planning/obstacles/urdf/obstacles.urdf',
                          [0.25, 0, -0.1], p.getQuaternionFromEuler([0, 0, 0]))



        rrt_tree = rrt(start, goal)
        if rrt_tree:
            path = reconstruct_path(rrt_tree, goal)
            print("Path:", path)
            np.savetxt('planning/trajectory/fcl_%d.csv'%int(time.time()),path)


        else:
            print("Path not found!")

