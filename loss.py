import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()

        self.loss = {}
        index = 0
        #self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_name_type = loss.split('*')
            loss_name,loss_type = loss_name_type.split('_')

            if loss_type == 'MSE':
                loss_function = nn.MSELoss(reduction='sum')
            elif loss_type == 'L1':
                loss_function = nn.L1Loss(reduction='sum')
            elif loss_type == 'TV':
                loss_function = TVLoss()
            else:
                assert False, f"Unsupported loss type: {loss_type:s}"
            if float(weight)>0:
                self.loss[loss_name]={
                    'name': loss_name,
                    'type': loss_type,
                    'index': index,
                    'weight': float(weight),
                    'function': loss_function}
                index=index+1
                

        if len(self.loss.keys()) > 1:
            self.loss['All']={'name': 'All','type': 'Total', 'weight': 0, 'function': None,'index': index}

        for l in self.loss.keys():
            if self.loss[l]['function'] is not None:
                print('{} : {:.3f} * {}'.format(self.loss[l]['name'], self.loss[l]['weight'], self.loss[l]['type']))
                #self.loss_module.append(l['function'])

        self.log = torch.Tensor()
        

    def forward(self, loss_name, sr, hr=None, element_num=0):
        #print('loss name name:',loss_name)
        if self.loss[loss_name]['type']=='L1' or self.loss[loss_name]['type']=='MSE':
            if element_num==0:
                element_num=sr.shape[0]*sr.shape[1]*sr.shape[2]*sr.shape[3]
            if self.loss[loss_name]['name']=='perceptual':
                loss = self.loss[loss_name]['function'](sr, hr)/(element_num+1e-20)
            else:
                loss = self.loss[loss_name]['function'](sr/255.0, hr/255.0)/(element_num+1e-20)
        
        if self.loss[loss_name]['type']=='TV':
            loss = self.loss[loss_name]['function'](sr/255.0) 

        effective_loss = self.loss[loss_name]['weight'] * loss
        
        
        self.log[-1, self.loss[loss_name]['index']] += effective_loss.item()
        self.log[-1, -1] += effective_loss.item()

        return effective_loss

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss.keys()))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(self.loss[l]['name'], c / n_samples))
        
        return ''.join(log)

    def plot_loss(self, apath, epoch): #画的是每个epoch 每种带了权重的loss即effective_loss变化曲线
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(self.loss[l]['name'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, self.loss[l]['index']].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, self.loss[l]['name']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

if __name__ == '__main__':
    import cv2
    import numpy as np
    gradient_nopad=Get_gradient_nopadding()
    gradient = Get_gradient()
    imgpath = '/home/datalab/ex_disk1/open_dataset/SR_data/test/Set14/monarch.png'
    image = cv2.imread(imgpath)
    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image = torch.from_numpy(image.copy())
    image = image.permute((2, 0, 1))
    image = torch.unsqueeze(image, dim=0).cuda()
    img_gradient_nopad = gradient_nopad(image)
    img_gradient = gradient(image)
    cv2.imwrite('3_gradient_nopad.jpg',np.uint8(img_gradient_nopad.squeeze(0).permute((1, 2, 0)).data.cpu().numpy()[:,:,::-1]))
    cv2.imwrite('3_gradient.jpg',np.uint8(img_gradient.squeeze(0).permute((1, 2, 0)).data.cpu().numpy()[:,:,::-1]))