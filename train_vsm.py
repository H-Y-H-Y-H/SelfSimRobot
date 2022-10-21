import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import torchvision
from query_model import QueryModel
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time

# Check GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("start", device)

DATA_PATH = "/Users/jionglin/Downloads/vsm/vsm_data_03/img/"
files = os.listdir(DATA_PATH)
f_train, f_test = train_test_split(files, test_size=0.2)

Angle_list_train = []
Angle_list_test = []
for f in f_train:
    angles = np.array([float(f.split("_")[0]),
                       float(f.split("_")[1]),
                       float(f.split("_")[2].split(".")[0] + "." + f.split("_")[2].split(".")[1])])
    Angle_list_train.append(angles)

for f in f_test:
    angles = np.array([float(f.split("_")[0]),
                       float(f.split("_")[1]),
                       float(f.split("_")[2].split(".")[0] + "." + f.split("_")[2].split(".")[1])])
    Angle_list_test.append(angles)

print(len(Angle_list_train), len(Angle_list_test))

def add_z(input_data):
    add_z_data = torch.tensor([])
    for i in range(len(input_data)):
        for z in range(128):
            vol = torch.cat((input_data[i], torch.tensor([z])))
            vol = torch.unsqueeze(vol, dim=0)
            add_z_data = torch.cat((add_z_data, vol))
    return add_z_data

def image_to_data(image, angle):
    data = torch.tensor([])
    label = torch.tensor([])
    for x in range(64-10, 64+10):
        for y in range(64-10, 64+10):
            one_label = np.array([image[0][x][y]])
            one_label = torch.from_numpy(one_label).to(device, dtype=torch.float)
            label = torch.cat((label, one_label))
            for z in range(128):
                vol = torch.cat((angle, torch.tensor([[x, y, z]])), dim=1)
                # vol = torch.unsqueeze(vol, dim=0)
                data = torch.cat((data, vol))
    return data, label


class PredData(Dataset):
    def __init__(self, angle_list, file_list):
        self.a_list = angle_list
        self.f_list = file_list
        pass

    def __getitem__(self, idx):
        # idx between 0 and 488095744
        # img_id = idx // (128 * 128)  # get the floor div by img size
        # img_rest = idx % (128 * 128)
        # x_id = img_rest // 128  # get the floor, img-rest div by img length, as row number
        # y_id = img_rest % 128  # get the rest, as column number
        # img = np.loadtxt(DATA_PATH + self.f_list[img_id])
        # input_data_sample = np.concatenate((self.a_list[img_id], np.array([x_id, y_id])))
        # label_data_sample = np.array([img[x_id, y_id]])
        # input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        # label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        # sample = {"input": input_data_sample, "label": label_data_sample}

        img = np.loadtxt(DATA_PATH + self.f_list[idx])
        sample = {"img": img, "angle": self.a_list[idx]}
        return sample

    def __len__(self):
        # return len(self.a_list) * 128 * 128
        return len(self.a_list)


def train_model(model, batch_size, lr, num_epoch, log_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = PredData(angle_list=Angle_list_train, file_list=f_train)
    test_dataset = PredData(angle_list=Angle_list_test, file_list=f_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf

    for epoch in range(num_epoch):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for i, bundle in enumerate(train_dataloader):
            # input_d, label_d = bundle["input"], bundle["label"]
            # input_d = add_z(input_d)
            image, angle = bundle["img"], bundle["angle"]
            input_d, label_d = image_to_data(image, angle)

            pred_result = model.forward(input_d.float())

            loss = model.loss(pred_result, label_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_l.append(loss.item())
            # running_loss += loss.item()
            print("training loss: ", loss.item())
            # if i % 100 == 99:    # print every 200 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.6f}')
            #     running_loss = 0.0

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()
        temp_l = []

        with torch.no_grad():
            print("...")
            for i, bundle in enumerate(test_dataloader):
                # input_d, label_d = bundle["input"], bundle["label"]
                # input_d = add_z(input_d)
                image, angle = bundle["img"], bundle["angle"]
                input_d, label_d = image_to_data(image, angle)
                pred_result = model.forward(input_d)
                loss = model.loss(pred_result, label_d)
                temp_l.append(loss.item())

            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)

        if test_mean_loss < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "training mean loss: ",train_mean_loss, "lr:", lr)
        print(epoch, "training loss: ", train_mean_loss, "Test loss: ", test_mean_loss)


if __name__ == "__main__":
    Lr = 1e-6
    Batch_size = 1  # 128
    b_size = 400
    Num_epoch = 100

    Log_path = "./log_01/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass

    Model = QueryModel(bsize=b_size).to(device)
    train_model(model=Model, batch_size=Batch_size, lr=Lr, num_epoch=Num_epoch, log_path=Log_path)

    pass
