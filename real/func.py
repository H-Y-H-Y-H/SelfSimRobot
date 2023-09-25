import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
list_img = []
# process image:
def median_img():

    for i in range(0,166855,500):
        print(i)
        img = plt.imread('../../FBVSM_DATA/real_data1/%06d.png'%i )
        img = cv2.resize(img,(100,100))
        list_img.append(img)
    med_img = np.median(list_img,axis=0)
    plt.imsave('median_img.png',med_img)

# median_img()

def mean_img():
    for i in [125,56597,157533,157720]:
        print(i)
        img = plt.imread('../../FBVSM_DATA/real_data1/%06d.png' % i)
        list_img.append(img)
        mean_img = np.mean(list_img,axis=0)

    img1 = plt.imread('../../FBVSM_DATA/real_data1/%06d.png' % 43152)
    img_k = np.mean([mean_img, img1, mean_img,mean_img], axis=0)
    plt.imsave('mean_img0.png', img_k)

    # # plt.imshow(mean_img)
    # # plt.show()
    # img1 = plt.imread('../../FBVSM_DATA/real_data1/%06d.png'%3900)
    # # #
    # img0 = plt.imread('mean_img%d.png'%0)[...,:3]
    # # img2 = plt.imread('mean_img%d.png'%2)[...,:3]
    # #
    # img_k = np.mean([img0,img1,img0], axis=0)
    # plt.imsave('mean_img99.png', img_k)
# mean_img()
# quit()


# real image
def occ_img():

    padding = 40
    med_img = plt.imread('robo0/median_img.png')[...,:3]
    height=med_img.shape[0]
    med_img[height-padding:,:,:] = [1,1,1]
    med_img[:padding,:,:] = [1,1,1]
    med_img[:,height-padding:,:] = [1,1,1]
    med_img[:,:padding,:] = [1,1,1]

    mask = (med_img < 0.2).any(axis=-1)
    output = np.zeros_like(med_img)
    output[mask] = [1, 1, 1]

    # plt.imsave('occ_img.png', output)
    return output


# output = occ_img()
# plt.imshow(output)
# plt.show()
# quit()

from PIL import Image


def change_ee_img2bool():


    img_bg = plt.imread('median_img.png')[..., :3]
    # img_bg = np.where(med_img_occ == 1., 1.,img_bg)

    for i in range(166855):
        print(i)
        img1 = plt.imread('../../FBVSM_DATA/real_data1/%06d.png'%i)
        img1 = cv2.resize(img1,(100,100))
        img1 = np.where(img1[...,2:]<0.2, img_bg,img1)
        img1 = np.where(img1[...,:1]>0.16, img_bg,img1)


        img_diff = img_bg - img1
        threshold_value = 0.1
        mask = (img_diff > threshold_value).any(axis=-1)

        # mask_black = (img_diff > threshold_value).any(axis=-1)

        # Create an output array of the same shape as the input image, initialized to black
        output = np.zeros_like(img_diff)

        # Set the pixels where the mask is True to white
        output[mask] = [255, 255, 255]

        binary_img = Image.fromarray(output.astype(np.uint8))

        binary_img.save("../../FBVSM_DATA/real_binary_data1(ee)/%06d.png"%i)
        # plt.imshow(output)
        # plt.show()
        # break
# change_ee_img2bool()


def remove_the_robot_root():
    data_root = '../../FBVSM_DATA/real_binary_data1/'
    save_root = '../../FBVSM_DATA/real_robo1/'
    os.makedirs(save_root,exist_ok=True)
    files_list = os.listdir(data_root)
    base_img = plt.imread('robo0/occ_img.png')[...,:3]
    for i in range(len(files_list)):
        img = plt.imread(data_root+'%06d.png'%i)[...,:3]
        img = img - base_img
        img = np.clip(img,0,1)
        # plt.imshow(img)
        # plt.show()
        plt.imsave(save_root+'%06d.png'%i,img)
        # break


def get_arm_data():
    save_pth = "../../FBVSM_DATA/real_bimg_robo0/"
    os.makedirs(save_pth,exist_ok=True)
    num_img = 138537 #166855
    TASK = 13
    # for i in range(TASK*10000,(TASK+1)*10000):#166855
    for i in range(300):
        print(i)
        image = plt.imread('../../FBVSM_DATA/real_data0/%06d.png'%i )[...,:3]
        image_g = np.where(image[...,1:2]<0.08, image[...,1:2],0)
        image_g[:, :22] = 0


        image_g = cv2.resize(image_g,(100,100))

        # image_b = np.where(image[...,2:]>0.2,   image[...,2:], 0)
        # img = np.dstack((image_r,image[...,1:2],image_b))
        image_g  = np.where(image_g > 0.,1, 0)
        image_g[60:] = 0

        # bool_img = np.zeros_like(image_g)
        # bool_img[r_indices] = 1
        # binary_img = Image.fromarray(bool_img.astype(np.uint8))
        # binary_img.save(save_pth + '/%06d.png' % i)

        # Blue for the robo 1
        # image = cv2.resize(image,(100,100))
        # image_r = np.where(image[...,:1] < 0.15,  1, 0)
        # image_b = image[...,2:]*image_r[...,:1]
        # image_b = np.where(image_b[...,:1] > 0.15,  1, 0)
        # image_b = image_b.reshape(100,100)
        # image_b[60:,50:] = 0

        plt.imsave(save_pth + '/%06d.png'%i,image_g, cmap='gray')

        # plt.imshow(image_g)
        # plt.show()



# get_arm_data()



def create_npz_data():
    cmds_angle = np.loadtxt('../data/real_data/log_pos_robo0.csv')*90
    cmds_angle[:,1:] *= -1
    num_img = len(cmds_angle)
    all_real_img = []
    focal = 130.2544532346901
    # pre_data = np.load('../data/real_data/real_data0920_robo1_166855.npz')

    for i in range(num_img):

        print(i)
        all_real_img.append(plt.imread('../../FBVSM_DATA/real_bimg_robo0/%06d.png'%i)[...,0])
    np.savez('../data/real_data/real_data0920_robo0_%d.npz' % (num_img),
             images=all_real_img,
             focal =focal,
             angles=cmds_angle)

create_npz_data()
