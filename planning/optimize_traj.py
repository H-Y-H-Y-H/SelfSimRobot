from visualize_bullet import *


def is_collision_line(point1, point2, num_samples=10):
    for i in np.linspace(0, 1, num_samples):
        interpolated_point = tuple(a * (1 - i) + b * i for a, b in zip(point1, point2))
        if is_collision(interpolated_point):
            return True
    return False


def shortcut_path(path):
    i = 0
    while i < len(path) - 2:
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


from scipy.interpolate import BSpline, make_interp_spline

def smooth_path_bspline(path, degree=3):
    path_array = np.array(path)
    t = range(path_array.shape[0])
    splines = [make_interp_spline(t, path_array[:,i], k=degree) for i in range(path_array.shape[1])]
    alpha = np.linspace(0, len(path) - 1, 100)  # Change 100 based on the resolution you need
    smoothed_path = np.vstack([spl(alpha) for spl in splines]).T
    return smoothed_path.tolist()


if __name__ == "__main__":
    DOF = 4
    robot_id = 1
    EndeffectorOnly = False
    seed = 0
    action_space = 90

    if robot_id == 0:
        data_point = 138537
    else:
        data_point = 166855



    test_name_ee = 'real_train_1_log0928_%ddof_%d_ee(%d)/' % (data_point, 100, seed)
    test_model_ee_pth =  '../train_log/%s/best_model/' % test_name_ee

    test_name = 'real_train_1_log0928_%ddof_%d(%d)/' % (data_point, 100, seed)
    test_model_pth = '../train_log/%s/best_model/' % test_name

    # DOF + 3 -> xyz and angle2 or 3 -> xyz
    model, optimizer = init_models(d_input=(DOF - 2) + 3,
                                   n_layers=4,
                                   d_filter=128,
                                   skip=(1, 2),
                                   output_size=2)

    model_ee, _ = init_models(d_input=(DOF - 2) + 3,
                                   n_layers=4,
                                   d_filter=128,
                                   skip=(1, 2),
                                   output_size=2)

    model.load_state_dict(torch.load(test_model_pth + "best_model.pt", map_location=torch.device(device)))
    model = model.to(torch.float64)
    model.eval()

    model_ee.load_state_dict(torch.load(test_model_ee_pth + "best_model.pt", map_location=torch.device(device)))
    model_ee = model_ee.to(torch.float64)
    model_ee.eval()

