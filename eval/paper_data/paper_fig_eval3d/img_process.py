import matplotlib.pyplot as plt
import numpy as np


w=2560
h=1440
c_w= 1280
c_h= 960

w1 = int((w-c_w)/2)
w2 = int((w+c_w)/2)
h1 = int((h-c_h)/2)
h2 = int((h+c_h)/2)

color_select = np.asarray([0,1,0])
save_path = 'robot0_shadow/'
for i in range(500):
    img = plt.imread('%d.png'%i)

    img = img[h1:h2,w1:w2,:3]
    dist = np.sum((img -color_select)**2,axis=2)
    print(dist.shape)
    mask = np.where(dist<0.4,1,0)
    img_mask = np.dstack((mask,mask,mask,mask*0.5))

    # plt.imshow(img_mask)
    # plt.show()
    plt.imsave(save_path+'%d.png'%i,img_mask)

