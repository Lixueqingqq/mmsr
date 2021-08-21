#-*- coding:utf-8
import os
import sys
import torch
BASIDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASIDIR)

from model import drn
import argparse
import numpy as np
import cv2
import collections
import time
import math
from skimage.measure import compare_ssim

from split_patch import split_image_into_overlapping_patches, stich_together


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class DRN_Param():
    def __init__(self, cpu, scale, rgb_range, n_colors, model, n_blocks, n_feats, negval, self_ensemble, pre_train,up_type,add_gradient_branch):
        self.scale = scale
        self.cpu = cpu
        self.rgb_range = rgb_range
        self.n_colors = n_colors
        self.model = model
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.negval = negval
        self.self_ensemble = self_ensemble
        self.pre_train = pre_train
        self.up_type = up_type
        self.add_gradient_branch = add_gradient_branch
    def print(self):
        print('args: scale:{}, cpu:{}, negval:{}, self_ensemble:{}, rgb_range:{}, n_colors:{}, model:{}, n_blocks:{}, n_feats:{}, pre_train:{},up_type:{},add_gradient_branch:{}'.format
            (self.scale, self.cpu, self.negval, self.self_ensemble, self.rgb_range, self.n_colors, self.model, self.n_blocks, self.n_feats, self.pre_train,self.up_type,self.add_gradient_branch))

class DRNModel():
    def __init__(self):
        para_dict = {}
        para_dict['cpu'] = False
        #para_dict['pre_train'] = "./pretrained/DRNS4x.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_supervised_bic/model/model_best.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4/model/model_best.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_supervised_bic_near_perceptual/model/model_best.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_bic-near/model/model_best.pt"
        #para_dict['pre_train'] = '/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_bic_perceptual/model/model_best.pt'
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_perceptual_bic-near10/model/model_latest.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_perceptual_bic-near_gradient_upconv/model/model_latest.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_perceptual_bic-near_upconv_gradientbranch/model/model_latest.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_bic_gradientbranch/model/model_latest.pt"
        #para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_perceptual_bic_gradientbranch/model/model_latest.pt"
        para_dict['pre_train'] = "/home/datalab/ex_disk2/moyan/SR/super_resolution_mmsr/log/DRN-Sx4_unsupervised_perceptual_bic_gradientbranch1/model/model_latest.pt"
        para_dict['model'] = "DRN-S" ##DRN-S | DRN-L
        para_dict['scale'] = 4
        para_dict['rgb_range'] = 255
        para_dict['n_colors'] = 3
        para_dict['n_feats'] = 16
        para_dict['n_blocks'] = 30
        para_dict['negval'] = 0.2
        para_dict['self_ensemble'] = True
        para_dict['up_type'] = 'pixelshuffle'
        para_dict['add_gradient_branch'] = False
        opt = DRN_Param(**para_dict)
        opt = self.init_model(opt)
        opt.scale = [pow(2, s+1) for s in range(int(np.log2(opt.scale)))]

        if 'upconv' in opt.pre_train:
            opt.up_type = 'upconv'
        
        if 'gradientbranch' in opt.pre_train:
            opt.add_gradient_branch = True

        opt.print()

        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = drn.make_model(opt).to(self.device)
        self.load(pre_train=opt.pre_train, cpu=opt.cpu)
        self.model = self.model.eval()
        if self.opt.add_gradient_branch:
            self.max_side = 320
        else:
            self.max_side = 800


    def init_model(self, args):
        # Set the templates here
        if args.model.find('DRN-S') >= 0:
            if args.scale == 4:
                args.n_blocks = 30
                args.n_feats = 16
            elif args.scale == 8:
                args.n_blocks = 30
                args.n_feats = 8
            else:
                print('Use defaults n_blocks and n_feats.')
            args.dual = True

        if args.model.find('DRN-L') >= 0:
            if args.scale == 4:
                args.n_blocks = 40
                args.n_feats = 20
            elif args.scale == 8:
                args.n_blocks = 36
                args.n_feats = 10
            else:
                print('Use defaults n_blocks and n_feats.')
            args.dual = True
        return args


    def load(self, pre_train, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )

    def predict(self, image):
        image = image.astype(np.float32)
        image = image[:, :, ::-1]
        image = torch.from_numpy(image.copy())
        image = image.permute((2, 0, 1))
        image = torch.unsqueeze(image, dim=0)
        if not self.cpu:
            image = image.cuda()

        #torch.cuda.synchronize()
        #st = time.time()
        with torch.no_grad():
            if self.opt.add_gradient_branch:
                output,gd_output = self.model(image)
            else:
                output = self.model(image)
        #torch.cuda.synchronize()
        #print('model infer time', time.time() - st, image.shape)

        outputx2 = output[1]
        outputx2 = torch.squeeze(outputx2)
        outputx2 = torch.clamp(outputx2, 0, 255)
        outputx2 = outputx2.permute((1, 2, 0))
        outputx2  = outputx2.data.cpu().numpy().copy().astype('uint8')
        outputx2 = outputx2[:, :, ::-1]
       
        outputx4 = output[2]
        outputx4 = torch.squeeze(outputx4)
        outputx4 = torch.clamp(outputx4, 0, 255)
        outputx4 = outputx4.permute((1, 2, 0)).data.cpu().numpy().astype('uint8')
        outputx4 = outputx4[:, :, ::-1]
        torch.cuda.empty_cache()
        return [outputx2, outputx4]


    def predict_patch(self, lr_img, by_patch_of_size=300):
        #padding_size= int(0.1 * by_patch_of_size) #10
        padding_size=10
        patches, p_shape = split_image_into_overlapping_patches(
            lr_img, patch_size=by_patch_of_size, padding_size=padding_size
        )
        print("predict path num:{} patch_size:{}".format(len(patches), patches[0].shape[0]))

        collectx2  = None
        collectx4 =  None
        for i in range(0, len(patches)):
            batch = self.predict(patches[i])
            batchx2 = np.expand_dims(batch[0], axis=0)
            batchx4 = np.expand_dims(batch[1], axis=0)
            if i == 0:
                collectx2 = batchx2
                collectx4 = batchx4
            else:
                collectx2 = np.append(collectx2, batchx2, axis=0)
                collectx4 = np.append(collectx4, batchx4, axis=0)
           
        scale = 2
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_img.shape[0:2], scale)) + (3,)
        upx2 = stich_together(
            collectx2,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding_size * scale,)
       #print(upx2.shape)
        scale = 4
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_img.shape[0:2], scale)) + (3,)
        upx4 = stich_together(
            collectx4,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding_size * scale,)
        #print(upx4.shape)
        return [upx2, upx4]

    def predict_adj(self, image):
        h,w = image.shape[0:2]
        start_time = time.time()
        if h > self.max_side or w > self.max_side:
            print("sr in patches")
            out = self.predict_patch(image)
        else:
            print("sr in whole pic")
            out = self.predict(image)
        print("run sr net time :{:.3f}ms".format((time.time() - start_time) * 1000))
        return out

        

def resize_image(image):
    max_side = 512.0
    ratio = min(max_side/image.shape[0], max_side/image.shape[1])
    nh = int(ratio * image.shape[0])
    nw = int(ratio * image.shape[1])
    s_image = cv2.resize(image, (nw, nh))
    return s_image

def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2, multichannel=True):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    return compare_ssim(img1, img2, multichannel=multichannel)


drn_model = DRNModel()
import glob
#image_paths = glob.glob("/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/bic/bicLRx4/*.png")
image_paths =  glob.glob('/home/datalab/ex_disk2/moyan/data/SR_data/valid_bsd/*.png')
#image_paths = glob.glob('/home/datalab/ex_disk1/open_dataset/SR_data/test/General100/*.png')
#image_paths = glob.glob('/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/near/nearLRx4/*.png')
#hr_path = '/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/HR/DIV2K_valid_HR'
hr_path = '/home/datalab/ex_disk2/moyan/data/SR_data/BSDS100'
#result_path = 'supervised_bic_near_perceptual_result_near'
#result_path = 'unsupervised_bic-near_result_near'
result_path = os.path.join(os.path.abspath(os.path.join(drn_model.opt.pre_train, "../..")),'predict')
if not os.path.exists(result_path):
    os.makedirs(result_path)
psnr_mean = 0
ssim_mean = 0
need_hr = True
for i in range(0, len(image_paths)):
    image_path = image_paths[i]
    #image_path = '/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/bic/bicLRx4/0855.png'
    print(image_path)
    image = cv2.imread(image_path)
    #image = cv2.resize(image,(320,320))
    print('orishapeeeeeeee:',image.shape)
    if max(image.shape[0],image.shape[1])>900:
        continue
    image_name = str(image_path).split('/')[-1].split('.')[0]
    s_image = image
    #cv2.imwrite("../../images/temp.jpg", s_image)

    h,w = s_image.shape[0:2]
   
    #x2 = cv2.resize(s_image,  (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
    #x4 = cv2.resize(s_image, (4*w, 4*h), interpolation=cv2.INTER_CUBIC)
    
    result = drn_model.predict_adj(s_image)
    print('result shape:',result[0].shape,result[1].shape)
    #x2 = np.concatenate((x2, result[0]), axis=1)
    if need_hr:
        hr_image = cv2.imread(os.path.join(hr_path,image_name.split('_')[0]+'.png'))
        print('shapeeeee:',hr_image.shape,4*h,4*w)
        hr_image = hr_image[:4*h,:4*w,:]
        psnrr = psnr(hr_image,result[1])
        ssimr = ssim(hr_image,result[1])
        psnr_mean = psnr_mean + psnrr
        ssim_mean = ssim_mean + ssimr
        x4 = np.concatenate((hr_image, result[1]), axis=1)
    else:
        x4 = result[1]

    #cv2.imwrite(f"result_patchimg_mmsr/{image_name}_x2.jpg", x2)
    cv2.imwrite(result_path+'/'+image_name+'_x4.jpg', x4)

print('the mean ssim is:', ssim_mean/len(image_paths))
print('the mean psnr is:', psnr_mean/len(image_paths))
############################################################################
image_paths = glob.glob('/home/datalab/ex_disk1/open_dataset/SR_data/test/General100/*.png')
psnr_mean = 0
ssim_mean = 0
need_hr = False
for i in range(0, len(image_paths)):
    image_path = image_paths[i]
    #image_path = '/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/bic/bicLRx4/0855.png'
    print(image_path)
    image = cv2.imread(image_path)
    #image = cv2.resize(image,(320,320))
    print('orishapeeeeeeee:',image.shape)
    if max(image.shape[0],image.shape[1])>900:
        continue
    image_name = str(image_path).split('/')[-1].split('.')[0]
    s_image = image
    #cv2.imwrite("../../images/temp.jpg", s_image)

    h,w = s_image.shape[0:2]
   
    #x2 = cv2.resize(s_image,  (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
    #x4 = cv2.resize(s_image, (4*w, 4*h), interpolation=cv2.INTER_CUBIC)
    
    result = drn_model.predict_adj(s_image)
    print('result shape:',result[0].shape,result[1].shape)
    #x2 = np.concatenate((x2, result[0]), axis=1)
    if need_hr:
        hr_image = cv2.imread(os.path.join(hr_path,image_name.split('_')[0]+'.png'))
        print('shapeeeee:',hr_image.shape,4*h,4*w)
        hr_image = hr_image[:4*h,:4*w,:]
        psnrr = psnr(hr_image,result[1])
        ssimr = ssim(hr_image,result[1])
        psnr_mean = psnr_mean + psnrr
        ssim_mean = ssim_mean + ssimr
        x4 = np.concatenate((hr_image, result[1]), axis=1)
    else:
        x4 = result[1]

    #cv2.imwrite(f"result_patchimg_mmsr/{image_name}_x2.jpg", x2)
    cv2.imwrite(result_path+'/'+image_name+'_x4.jpg', x4)

print('the mean ssim is:', ssim_mean/len(image_paths))
print('the mean psnr is:', psnr_mean/len(image_paths))

image_paths = glob.glob("/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/bic/bicLRx4/*.png")
hr_path = '/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/HR/DIV2K_valid_HR'
psnr_mean = 0
ssim_mean = 0
need_hr = True
for i in range(0, len(image_paths)):
    image_path = image_paths[i]
    #image_path = '/home/datalab/ex_disk1/open_dataset/SR_data/DIV2K/LR/valid/bic/bicLRx4/0855.png'
    print(image_path)
    image = cv2.imread(image_path)
    #image = cv2.resize(image,(320,320))
    print('orishapeeeeeeee:',image.shape)
    if max(image.shape[0],image.shape[1])>900:
        continue
    image_name = str(image_path).split('/')[-1].split('.')[0]
    s_image = image
    #cv2.imwrite("../../images/temp.jpg", s_image)

    h,w = s_image.shape[0:2]
   
    #x2 = cv2.resize(s_image,  (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
    #x4 = cv2.resize(s_image, (4*w, 4*h), interpolation=cv2.INTER_CUBIC)
    
    result = drn_model.predict_adj(s_image)
    print('result shape:',result[0].shape,result[1].shape)
    #x2 = np.concatenate((x2, result[0]), axis=1)
    if need_hr:
        hr_image = cv2.imread(os.path.join(hr_path,image_name.split('_')[0]+'.png'))
        print('shapeeeee:',hr_image.shape,4*h,4*w)
        hr_image = hr_image[:4*h,:4*w,:]
        psnrr = psnr(hr_image,result[1])
        ssimr = ssim(hr_image,result[1])
        psnr_mean = psnr_mean + psnrr
        ssim_mean = ssim_mean + ssimr
        x4 = np.concatenate((hr_image, result[1]), axis=1)
    else:
        x4 = result[1]

    #cv2.imwrite(f"result_patchimg_mmsr/{image_name}_x2.jpg", x2)
    cv2.imwrite(result_path+'/'+image_name+'_x4.jpg', x4)

print('the mean ssim is:', ssim_mean/len(image_paths))
print('the mean psnr is:', psnr_mean/len(image_paths))