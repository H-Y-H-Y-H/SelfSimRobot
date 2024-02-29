import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

robot_id = 2

w=2560
h=1440
c_w= 580
c_h= c_w

w1 = int((w-c_w)/2)
w2 = int((w+c_w)/2)
h1 = int((h-c_h)/2)
h2 = int((h+c_h)/2)
arm_ee = 'arm'

save_path = 'robot%d_shadow(%s)/' % (robot_id,arm_ee)
os.makedirs(save_path, exist_ok=True)

def generate_frames():
    color_select = np.asarray([0, 1, 0])
    alpha = 0.5

    for i in range(2000):
        print(i)
        real_robot_img = np.copy(plt.imread('real_results/1026_r%d/%d.jpeg'%(robot_id,i)).astype(float))/255

        img = plt.imread('robot%d_sim(%s)/%d.jpeg'%(robot_id,arm_ee,i))/255
        # img = plt.imread('robot%d_sim/%d.png'%(robot_id,i))[...,:3]

        img = img[h1:h2,w1:w2,:3]
        img = cv2.resize(img,(720,720))

        dist = np.sum((img -color_select)**2,axis=2)
        print(dist.shape)
        mask = np.where(dist<0.4,1.,0.)
        zero_mask = np.zeros_like(mask)
        img_mask = np.dstack((zero_mask,mask,zero_mask)).astype(float)
        real_robot_img += img_mask*alpha
        real_robot_img = np.clip(real_robot_img,0,1)
        # plt.imshow(real_robot_img)
        # plt.show()
        plt.imsave(save_path+'%d.jpeg'%i,real_robot_img)


generate_frames()


def frames_2_video():
    img_array = []
    for i in range(2000):
        filename = save_path+'/%d.jpeg'%(i)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('robot_%d(%s).avi'%(robot_id,arm_ee), cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


frames_2_video()

