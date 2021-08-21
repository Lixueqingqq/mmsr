import os
import cv2
import random
import shutil
import numpy as np
'''
place2_path = '/home/datalab/ex_disk1/open_dataset/AIDesignDataSets/places2/train_large'

select_place2_image_save_path = '/home/datalab/ex_disk1/open_dataset/SR_data/FPlace2'

folders = os.listdir(place2_path)

num=0

while num<3000:
    folder = random.choice(folders)

    sub_folder = random.choice(os.listdir(os.path.join(place2_path,folder)))

    image_select_path = random.choice(os.listdir(os.path.join(place2_path,folder,sub_folder)))

    image_path = os.path.join(place2_path,folder,sub_folder,image_select_path)
    
    file_name = 'fplace2_'+str(num)+'.jpg'
    
    if image_path.endswith('.jpg'):
        shutil.copyfile(image_path,os.path.join(select_place2_image_save_path,file_name))
        num=num+1
'''


#########concat result################

result_path = '/Users/lxq/work/SR/super_resolution_mmsr/log/DRN-Sx4/result_bic_predict_nearst_allimg_mmsr/'
set1_result_path = '/Users/lxq/work/SR/super_resolution_mmsr/log/DRN-Sx4_bic_perceptual/bic_perceptual_result_near/'
set2_result_path = '/Users/lxq/work/SR/super_resolution_mmsr/log/DRN-Sx4_supervised_bic_near_perceptual/supervised_bic_near_perceptual_result_near/'
set3_result_path = '/Users/lxq/work/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_bic-near/unsupervised_bic-near_result_near/'

#ESRGAN_path = '/Users/lxq/work/SR/super_resolution_mmsr/log/DRN-Sx4/result_ori_ESRGAN_up/'

for img in os.listdir(result_path):
    if img.endswith('_x4.jpg'):
        result = cv2.imread(os.path.join(set1_result_path,img))
        ori_img = result[:,:result.shape[1]//2,:]
        hr_mmsr = cv2.imread(os.path.join(result_path,img))[:,result.shape[1]//2:,:]
        set1_result = cv2.imread(os.path.join(set1_result_path,img))[:,result.shape[1]//2:,:]
        set2_result = cv2.imread(os.path.join(set2_result_path,img))[:,result.shape[1]//2:,:]
        set3_result = cv2.imread(os.path.join(set3_result_path,img))[:,result.shape[1]//2:,:]
        result = np.concatenate((ori_img,hr_mmsr, set1_result,set2_result,set3_result), axis=1)
        cv2.imwrite('/Users/lxq/work/SR/super_resolution_mmsr/log/set_results/'+'near_'+img,result)

