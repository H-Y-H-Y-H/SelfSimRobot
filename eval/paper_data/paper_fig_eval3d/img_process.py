import os

import matplotlib.pyplot as plt
import numpy as np
robot_id = 1

w=2560
h=1440
c_w= 1280
c_h= 1280

w1 = int((w-c_w)/2)
w2 = int((w+c_w)/2)
h1 = int((h-c_h)/2)
h2 = int((h+c_h)/2)

color_select = np.asarray([0,1,0])
save_path = 'robot%d_shadow/'%robot_id
os.makedirs(save_path,exist_ok=True)
for i in range(500):
    print(i)
    img = plt.imread('robot%d_sim/%d.png'%(robot_id,i))

    img = img[h1:h2,w1:w2,:3]
    plt.imsave('robot%d_sim/%d.jpeg'%(robot_id,i),img)

    # dist = np.sum((img -color_select)**2,axis=2)
    # print(dist.shape)
    # mask = np.where(dist<0.4,1.,0.)
    # img_mask = np.dstack((mask,mask,mask)).astype(float)

    # plt.imshow(img_mask)
    # plt.show()
    # plt.imsave(save_path+'%d.jpeg'%i,img_mask)
