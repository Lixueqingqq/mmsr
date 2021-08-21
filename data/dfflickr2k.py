import os
from data import srdata
import random
import glob

class DFFlickr2K(srdata.SRData):
    def __init__(self, args, name='DFFlickr2K', train=True, benchmark=False):
        super(DFFlickr2K, self).__init__(
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
        self.ext = ('.png', '.png')

        if self.train:
            apath = os.path.join(data_dir, 'Flickr2K')   
            dir_hr = os.path.join(apath, 'HR','Flickr2K_HR')
            dir_lr = os.path.join(apath, 'LR')
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
            
            names_hr = sorted(glob.glob(os.path.join(dir_hr, '*' + self.ext[0])))

            names_hrs.extend(names_hr)

            names_lr = [[] for _ in self.scale]
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                for si, s in enumerate(self.scale):
                    dir_lr_format = 'bic'
                    # if si == 0:
                    #      dir_lr_name = 'bic' if random.random() < 0.75 else "near"
                    
                    dir_lr_name = dir_lr_format + 'LRx' + str(s)
                    names_lr[si].append(os.path.join(
                        dir_lr, '{}/{}/{}{}'.format(
                            dir_lr_format,dir_lr_name, filename, self.ext[1]
                        )
                    ))
    
            
            for i in range(len(names_lr)):
                names_lrs[i].extend(names_lr[i])
        
        print("check data loader")
        for i in range(10):
            print(names_hrs[i], names_lrs[0][i], names_lrs[1][i])

        print(len(names_hrs), len(names_lrs[0]), len(names_lrs[1]))
            
        return names_hrs, names_lrs

