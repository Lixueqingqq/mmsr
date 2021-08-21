import os
from data import srdata
import random
import glob
import imageio
import cv2

class DFFlickr2K_Imagenet(srdata.SRData):
    def __init__(self, args, name='DFFlickr2K_Imagenet', train=True, benchmark=False):
        super(DFFlickr2K_Imagenet, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )


    def _set_filesystem(self, data_dir):
        apath = os.path.join(data_dir, 'DIV2K')
        self.dir_hr = []
        self.dir_lr = []
        dir_hr = os.path.join(apath, 'HR')
        dir_lr = os.path.join(apath, 'LR')
        self.dir_hr.append(dir_hr)
        self.dir_lr.append(dir_lr)

        apath = os.path.join(data_dir, 'Flickr2K')
        dir_hr = os.path.join(apath, 'HR')
        dir_lr = os.path.join(apath, 'LR')
        self.dir_hr.append(dir_hr)
        self.dir_lr.append(dir_lr)

        apath = os.path.join(data_dir, 'unsupervised')
        dir_hr = os.path.join(apath, 'LR/bicx4') # get the pair name
        dir_lr = os.path.join(apath, "LR") 
        self.dir_hr.append(dir_hr)
        self.dir_lr.append(dir_lr)   

        self.ext = ('.png', '.png')
        print('hr,lr dirs:', self.dir_hr, self.dir_lr)


    def _scan(self):
        names_hrs = []
        names_lrs = [[] for _ in self.scale]

        for i in range(len(self.dir_hr)):
            dir_hr = self.dir_hr[i]
            dir_lr = self.dir_lr[i]
            
            names_hr = sorted(
                glob.glob(os.path.join(dir_hr, '*' + self.ext[0]))
            )
            names_hrs.extend(names_hr)

            names_lr = [[] for _ in self.scale]
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                for si, s in enumerate(self.scale):
                    dir_lr_name = 'bic'
                    # if si == 0:
                    #      dir_lr_name = 'bic' if random.random() < 0.75 else "near"
                    
                    dir_lr_name = dir_lr_name + 'x' + str(s)
                    names_lr[si].append(os.path.join(
                        dir_lr, '{}/{}{}'.format(
                            dir_lr_name, filename, self.ext[1]
                        )
                    ))
            
            for i in range(len(names_lr)):
                names_lrs[i].extend(names_lr[i])
        
        print("check data loader")
        for i in range(10):
            print(names_hrs[i], names_lrs[0][i], names_lrs[1][i])
            
        return names_hrs, names_lrs


    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]

        filename = f_hr
        hr = imageio.imread(f_hr)
        hr = cv2.resize(hr, dsize=(4 * hr.shape[1], 4 * hr.shape[0]))
        lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]
        return lr, hr, filename

