import numpy as np
from env4 import FBVSM_Env
import pybullet as p
from train import *
from test_model import *
# changed Transparency in urdf, line181, Mar31

def interact_env():

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


def collision_free_planning():
    TASK = 0

    if TASK == 0: # circle



    # env.add_obstacles()

    obs = env.reset()
    c_angle = obs[0]
    debug_points = 0
    action_space = 90

    # input para
    motor_input = []
    for m in range(DOF):
        motor_input.append(p.addUserDebugParameter("motor%d:" % m, -1, 1, 0))

    for i in range(100000):
        for dof_i in range(DOF):
            c_angle[dof_i] = p.readUserDebugParameter(motor_input[dof_i])
        degree_angles = c_angle*action_space
        occu_pts = test_model(degree_angles,model)
        p_rgb = np.ones_like(occu_pts)
        p.removeUserDebugItem(debug_points)  # update points every step
        debug_points = p.addUserDebugPoints(occu_pts, p_rgb, pointSize=2)
        obs, _, _, _ = env.step(c_angle)


import heapq


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.g = float('inf')  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to goal
        self.parent = None  # Parent node in the path

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)


def a_star(grid, start, goal):
    open_list = [start]
    closed_list = set()
    start.g = 0
    start.h = abs(start.x - goal.x) + abs(start.y - goal.y) + abs(start.z - goal.z)

    while open_list:
        current = heapq.heappop(open_list)
        if current.x == goal.x and current.y == goal.y and current.z == goal.z:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(start)
            return path[::-1]

        closed_list.add((current.x, current.y, current.z))
        for dx, dy, dz in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]:
            x, y, z = current.x + dx, current.y + dy, current.z + dz
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and 0 <= z < len(grid[0][0]) and grid[x][y][z] == 0:
                neighbor = Node(x, y, z)
                if (x, y, z) in closed_list:
                    continue
                tentative_g = current.g + 1
                if neighbor not in open_list or tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = abs(neighbor.x - goal.x) + abs(neighbor.y - goal.y) + abs(neighbor.z - goal.z)
                    neighbor.parent = current
                    if neighbor not in open_list:
                        heapq.heappush(open_list, neighbor)
    return None  # No path found


# # Example Usage:
# grid = [
#     [
#         [0, 1, 0],
#         [0, 0, 0],
#         [0, 1, 0]
#     ],
#     [
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ],
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ]
# ]
# start = Node(0, 0, 0)
# goal = Node(2, 2, 2)
# path = a_star(grid, start, goal)
# if path:
#     for node in path:
#         print(node.x, node.y, node.z)
# else:
#     print("No path found!")


if __name__ == "__main__":
    DOF = 4
    robot_id = 0
    EndeffectorOnly = False
    seed = 0

    if robot_id == 0:
        data_point = 138537
    else:
        data_point = 166855


    if EndeffectorOnly:
        test_name = 'real_train_log_%ddof_%d(ee)(%d)/' % (data_point, 100, seed)
    else:
        test_name = 'real_train_log_%ddof_%d(%d)/' % (data_point, 100, seed)

    test_model_pth = 'train_log/%s/best_model/' % test_name


    model, optimizer = init_models(d_input=(DOF - 2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                   n_layers=4,
                                   d_filter=128,
                                   skip=(1, 2),
                                   output_size=2)

    model.load_state_dict(torch.load(test_model_pth + "nerf.pt", map_location=torch.device(device)))
    model = model.to(torch.float64)
    model.eval()

    # start simulation:
    p.connect(p.GUI)

    env = FBVSM_Env(
        show_moving_cam=False,robot_ID=robot_id,
        width=width,
        height=height,
        render_flag=True,
        num_motor=DOF,
        dark_background = True)


    interact_env()

    # collision_free_planning()
