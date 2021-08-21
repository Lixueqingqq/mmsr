import os
from data import srdata
import random
import glob
import imageio
import cv2
import numpy as np

class DFFlickr2K_Place(srdata.SRData):
    def __init__(self, args, name='DFFlickr2K_Place', train=True, benchmark=False):
        super(DFFlickr2K_Place, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        


    def _set_filesystem(self, data_dir):
        apath = os.path.join(data_dir, 'DIV2K')
        self.dir_hr = []
        self.dir_lr = []
        if self.train:
            self.split_name = 'train'
        else:
            self.split_name = 'valid'
        dir_hr = os.path.join(apath, 'HR','DIV2K_'+self.split_name+'_HR')
        dir_lr = os.path.join(apath, 'LR',self.split_name)
        self.dir_hr.append(dir_hr)
        self.dir_lr.append(dir_lr)
        #self.ext = ('.png', '.png')

        if self.train:
            apath = os.path.join(data_dir, 'Flickr2K')
            dir_hr = os.path.join(apath, 'HR','Flickr2K_HR')
            dir_lr = os.path.join(apath, 'LR')
            self.dir_hr.append(dir_hr)
            self.dir_lr.append(dir_lr)
            #self.ext = ('.png', '.png')
            #print('hr,lr dirs:', self.dir_hr, self.dir_lr)

        if self.add_unsupervised:
            apath = os.path.join(data_dir, 'FPlace2')
            #dir_hr = os.path.join(apath, 'HR','Flickr2k_HR') # get the pair name
            dir_hr = None
            dir_lr = os.path.join(apath, 'LR') 
            self.dir_hr.append(dir_hr)
            self.dir_lr.append(dir_lr)   

        self.ext = ('.png', '.png')
        print('hr,lr dirs:', self.dir_hr, self.dir_lr)


    def _scan(self):
        names_hrs = []
        names_lrs = [[] for _ in self.scale]
        sample_weight = []

        for i in range(len(self.dir_hr)):
            dir_hr = self.dir_hr[i]
            dir_lr = self.dir_lr[i]
            if dir_hr != None:
                names_hr = sorted(
                    glob.glob(os.path.join(dir_hr, '*' + self.ext[0]))
                )
                names_hrs.extend(names_hr)

                sample_weight.extend([0.7]*len(names_hr))

                names_lr = [[] for _ in self.scale]
                for f in names_hr:
                    filename, _ = os.path.splitext(os.path.basename(f))
                    for si, s in enumerate(self.scale):
                        dir_lr_format = 'bic'
                        if self.Degenerate_type == 'bic&near':
                            if si == 0:
                                dir_lr_format = 'bic' if random.random() < 0.75 else "near"
                        
                        dir_lr_name = dir_lr_format + 'LRx' + str(s)
                        names_lr[si].append(os.path.join(
                        dir_lr, '{}/{}/{}{}'.format(
                            dir_lr_format,dir_lr_name, filename, self.ext[1]
                        )
                        ))
            else:
                names_lr = [[] for _ in self.scale]
                dir_lr_format = 'bic'
                if self.train:
                    names_lr[-1] = sorted(
                        glob.glob(os.path.join(dir_lr.replace('LR','HR'),'FPlace2_HR'+str(int(self.scale[0]//2)),'*' + self.ext[0]))
                    )[:self.add_unsupervised_num_train]
                else:
                    names_lr[-1] = sorted(
                        glob.glob(os.path.join(dir_lr.replace('LR','HR'),'FPlace2_HR'+str(int(self.scale[0]//2)),'*' + self.ext[0]))
                    )[-self.add_unsupervised_num_valid:]

                names_hr = [None]*len(names_lr[-1])
                #print('tttttttttttt',len(names_lr[-1]),os.path.join(dir_lr.replace('LR','HR'),'FPlace2_HR'+str(int(self.scale[0]//2)),'*' + self.ext[0]))
                names_hrs.extend(names_hr)

                sample_weight.extend([0.3]*len(names_hr))

                #print('oooooooo:',len(names_hrs)) 

                for f in names_lr[-1]:
                    filename, _ = os.path.splitext(os.path.basename(f))
                    for si,s in enumerate(self.scale[:-1]):
                        if self.Degenerate_type == 'bic&near':
                            if si == 0:
                                dir_lr_format = 'bic' if random.random() < 0.75 else "near"  
                        dir_lr_name = dir_lr_format+'LRx'+str(int(s//2))
                        names_lr[si].append(os.path.join(dir_lr,'{}/{}/{}{}'.format(
                            dir_lr_format,dir_lr_name,filename,self.ext[1])
                        ))
                
                    
            for i in range(len(names_lr)):
                names_lrs[i].extend(names_lr[i])
        
        print("check data loader")
        for i in range(10):
            print(names_hrs[i], names_lrs[0][i], names_lrs[1][i])
        
        print(len(names_hrs), len(names_lrs[0]), len(names_lrs[1]),len(sample_weight))
        
        return names_hrs, names_lrs,sample_weight

    
    def _get_imgs_path(self, args):
        list_hr, list_lr,sample_weight = self._scan()
        
        #可直接在dataloader中设置shuffle=True，将数据打乱，没必要在这里打乱
        index = np.arange(len(list_hr))
        np.random.shuffle(index)
        self.images_hr, self.images_lr = [list_hr[i] for i in index], [[a[i] for i in index] for a in list_lr]
        self.sample_weight = [sample_weight[i] for i in index]
        #self.images_hr,self.images_lr = list_hr, list_lr
        #self.sample_weight = sample_weight


    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]
        lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]    

        if f_hr != None:
            filename, _ = os.path.splitext(os.path.basename(f_hr)) #获得文件名，对文件名拆分成名字和后缀
            hr = imageio.imread(f_hr)  ##读入的图是rgb类型的
        else:
            filename, _ = os.path.splitext(os.path.basename(f_lr[0])) 
            filename = 'unsupervised_'+ filename
            hr = np.zeros((lr[0].shape[0]*self.scale[0], lr[0].shape[1]*self.scale[0],3),dtype='uint8')
        
        return lr, hr, filename

     