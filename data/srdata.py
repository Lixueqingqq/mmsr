import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import albumentations as A
import random
import time

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.add_unsupervised = args.add_unsupervised
        self.Degenerate_type = args.Degenerate_type
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        self.add_unsupervised_num_train = 1480
        self.add_unsupervised_num_valid = 50
        
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()
        print('data_len is:',self.__len__())

        self.noise_aug = A.OneOf([
                            A.GaussNoise(var_limit=10.0, p=0.1),
                            A.IAAAdditiveGaussianNoise(scale=(0.005 * 255, 0.01 * 255), p=0.1),
                            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.1),
                            A.MotionBlur(blur_limit=5, p=0.05),
                            A.GaussianBlur(blur_limit=5, p=0.05)
                            ]
                        )
       
      
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr, hr = self.get_patch(lr, hr)   
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return self.dataset_length


    def _get_imgs_path(self, args):
        list_hr, list_lr = self._scan()
        '''
        #可直接在dataloader中设置shuffle=True，将数据打乱，没必要在这里打乱
        index = np.arange(len(list_hr))
        np.random.shuffle(index)
        self.images_hr, self.images_lr = [list_hr[i] for i in index], [[a[i] for i in index] for a in list_lr]
        '''
        self.images_hr,self.images_lr = list_hr, list_lr


    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = self.dataset_length // len(self.images_hr)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)


    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]

        filename, _ = os.path.splitext(os.path.basename(f_hr)) #获得文件名，对文件名拆分成名字和后缀
        hr = imageio.imread(f_hr)
        lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]
        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
            
            # add noise aug
            if self.args.add_noise:
                #ori_l = lr[0].copy()
                augmented = self.noise_aug(image=lr[0])
                lr[0] = augmented['image']  #只对目标低分辨率图片进行数据增强
                #print(lr[0].shape, lr[1].shape, hr.shape)
                '''
                if self.train and random.random()>0.5:
                    train_path_patch = './data_input_array'
                    if not os.path.exists(train_path_patch):
                        os.makedirs(train_path_patch)
                    imageio.imwrite('{}.png'.format(os.path.join(train_path_patch,str(time.time()))), np.concatenate((ori_l,lr[0]),axis=1))
                '''
        else:
            if isinstance(lr, list):
                ih, iw = lr[0].shape[:2]
            else:
                ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale[0], 0:iw * scale[0],:]
            
        return lr, hr
