import numpy as np
import matplotlib.pyplot as plt
from train_model import *


def check_training_plot():
    log_file = np.loadtxt("previous_files/log_03/training_MSE.csv")
    print(log_file.shape)

    ax = plt.subplot()
    # ax.plot([x for x in range(len(log_file))], log_file)
    ax.plot(log_file)
    plt.show()


def prepare_bullet():
    import pybullet as p

    RENDER = False
    if RENDER:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    env = FBVSM_Env()
    return env


def check_model():
    PATH = "previous_files/log_03/best_model_MSE.pt"
    model = VsmModel().to(device)
    model.load_state_dict(torch.load(PATH, map_location='cpu'))
    model.eval()

    env = prepare_bullet()
    l_array = np.linspace(-1.0, 1.0, num=21)
    t_angle = np.random.choice(l_array, 2)
    obs, _, _, _ = env.step(t_angle)
    a_list, img = obs[0], obs[1]
    # print(img)
    # print(a_list)
    img = (img / 255.).astype(np.float32)
    img = torch.Tensor(img).to(device)
    angles = torch.Tensor(a_list).to(device)

    xs = torch.randperm(400).reshape(4, 100).to(device)
    ys = torch.randperm(400).reshape(4, 100).to(device)
    for x in xs:
        for y in ys:
            # box_p = f_box.clone().to(device)
            # box_p = torch.index_select(box_p, 0, x).to(device)
            # box_p = torch.index_select(box_p, 1, y).to(device)
            # label = img.clone().to(device)
            # label = torch.index_select(label, 0, x).to(device)
            # label = torch.index_select(label, 1, y).to(device)
            #
            # d = ff.clone().to(device)
            # d = torch.index_select(d, 0, x).to(device)
            # d = torch.index_select(d, 1, y).to(device)
            f_box_t = torch.Tensor(f_box).to(device)
            box_p = f_box_t.clone().to(device)
            # box_p = f_box.clone().to(device)
            box_p = torch.index_select(box_p, 0, x).to(device)
            box_p = torch.index_select(box_p, 1, y).to(device)
            label = img.clone().to(device)
            label = torch.index_select(label, 0, x).to(device)
            label = torch.index_select(label, 1, y).to(device)

            # d = ff.clone().to(device)
            # d = torch.index_select(d, 0, x).to(device)
            # d = torch.index_select(d, 1, y).to(device)

            d = box_p[:, :, -1, :]
            # 400*400 rays and pixels, split into 100*100 rays and pixels, loop 16 times
            prediction = v_render(box_p, model, angles, d)
            print(prediction.shape)
            plt.imshow(prediction.cpu().detach().numpy())
            plt.show()


if __name__ == "__main__":
    check_training_plot()
    # check_model()
