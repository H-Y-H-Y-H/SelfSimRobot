import numpy as np
import matplotlib.pyplot as plt
import cv2
list_img = []
# process image:
def median_img():

    for i in range(0,138536,500):
        print(i)
        img = plt.imread('../../FBVSM_DATA/real_data/%06d.png'%i )
        img = cv2.resize(img,(100,100))
        list_img.append(img)
    med_img = np.median(list_img,axis=0)
    plt.imsave('median_img.png',med_img)

# median_img()
# quit()

def mean_img():
    for i in range(5000, 138536, 20000):
        print(i)
        img = plt.imread('../../FBVSM_DATA/real_data/%06d.png' % i)
        list_img.append(img)
        mean_img = np.mean(list_img,axis=0)
    plt.imsave('mean_img0.png', mean_img)

    # plt.imshow(mean_img)
    # plt.show()
    # img1 = plt.imread('../../FBVSM_DATA/real_data/%06d.png'%82410)
    # #
    # img0 = plt.imread('mean_img%d.png'%0)[...,:3]
    # img2 = plt.imread('mean_img%d.png'%2)[...,:3]
    #
    # img_k = np.mean([img0,img1,img2], axis=0)
    # plt.imsave('mean_img99.png', img_k)



# real image
def occ_img():

    padding = 40
    med_img = plt.imread('median_img.png')[...,:3]
    height=med_img.shape[0]
    med_img[height-padding:,:,:] = [1,1,1]
    med_img[:padding,:,:] = [1,1,1]
    med_img[:,height-padding:,:] = [1,1,1]
    med_img[:,:padding,:] = [1,1,1]

    mask = (med_img < 0.2).any(axis=-1)
    output = np.zeros_like(med_img)
    output[mask] = [1, 1, 1]

    plt.imsave('occ_img.png', output)
    return output


# output = occ_img()
# plt.imshow(output)
# plt.show()
# quit()

from PIL import Image
def change_img2bool():

    med_img_occ = occ_img()
    img_bg = plt.imread('median_img.png')[..., :3]
    img_bg = np.where(med_img_occ == 1., 1.,img_bg)

    for i in range(131609,133280):
        print(i)
        img1 = plt.imread('../../FBVSM_DATA/real_data/%06d.png'%i)
        img1 = cv2.resize(img1,(100,100))
        img1 = np.where(med_img_occ == 1., 0.,img1)


        img_diff = img_bg - img1
        threshold_value = 0.22
        mask = (img_diff > threshold_value).any(axis=-1)

        # Create an output array of the same shape as the input image, initialized to black
        output = np.zeros_like(img_diff)

        # Set the pixels where the mask is True to white
        output[mask] = [255, 255, 255]

        binary_img = Image.fromarray(output.astype(np.uint8))

        binary_img.save("../../FBVSM_DATA/real_binary_data/%06d.png"%i)
        # plt.imshow(output)
        # plt.show()
        # break


change_img2bool()

