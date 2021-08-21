import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from model import common
from option import args


def make_model(opt):
    if opt.add_gradient_branch:
        return DRN_Gradient(opt)
    else:
        return DRN(opt)


class DRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3
        up_type= opt.up_type
        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False, up_type=up_type),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False, up_type=up_type),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

   
    def forward(self, x, only_upsample=False):
        # upsample x to target sr size
        #print('xxxxxxxx:',x.shape)
        x = self.upsample(x)

        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        #print('down-----')
        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        #print('up-------')
        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        #print(sr.shape)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)
            #print(sr.shape)
            results.append(sr)
            
            if only_upsample:
                break

        return results

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
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
        

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        
        return x


class DRN_Gradient(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN_Gradient, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3
        up_type = opt.up_type

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False, up_type=up_type),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False, up_type=up_type),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

        #set gradient 
        self.get_g_nopadding = Get_gradient_nopadding()

        self.b_fea_conv = conv(opt.n_colors, n_feats * pow(2, self.phase), kernel_size=3)

        d_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p)*2, kernel_size, act=act
            ) for _ in range(3)
        ] for p in range(self.phase, 1, -1)
        ]

        d_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase)*2, kernel_size, act=act
            ) for _ in range(3)
        ])

        self.d_catconv =conv(n_feats * pow(2, self.phase)*2,n_feats * pow(2, self.phase),kernel_size=3)
       
        d_up = [[
            nn.Upsample(scale_factor=2,mode='nearest'),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        for p in range(self.phase - 1, 0, -1):
            d_up.append([
                nn.Upsample(scale_factor=2,mode='nearest'),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.d_sequnce_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.d_sequnce_blocks.append(
                nn.Sequential(*d_blocks[idx], *d_up[idx])
            )
        
        f_conv = [conv(n_feats * pow(2, self.phase-1)*3, n_feats * pow(2, self.phase), kernel_size=1)]
        for p in range(self.phase-1, 0, -1):
            f_conv.append(
                conv(n_feats * pow(2, p-1)*3, n_feats * pow(2, p), kernel_size=1)
            )
        self.f_conv = nn.ModuleList(f_conv)

        grad_tail = [conv(n_feats * pow(2, self.phase - 1), opt.n_colors, kernel_size)]
        for p in range(self.phase-1, 0, -1):
            grad_tail.append(
                conv(n_feats * pow(2, p-1), opt.n_colors, kernel_size)
            )
        self.grad_tail = nn.ModuleList(grad_tail)



    def forward(self, x, only_upsample=False):
        # upsample x to target sr size
        #print('xxxxxxxx:',x.shape)
        x_grad = self.get_g_nopadding(x)
        
        x = self.upsample(x)
        #print('uppppxxxxxxxx:',x.shape)
        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        #print('down-----')
        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        #print('up-------')
        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        #print(sr.shape)
        results = [sr]
        results_grad = []
        x_d_fea = self.b_fea_conv(x_grad)

        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx][0:10](x)
            x_fea1 = x
            x = self.up_blocks[idx][10:20](x)
            x_fea2 = x
            x = self.up_blocks[idx][20:30](x)
            x_fea3 = x
            x = self.up_blocks[idx][30:](x)
            
            x_cat = torch.cat([x_d_fea, x_fea1], dim=1) 
            x_cat = self.d_sequnce_blocks[idx][0](x_cat)
            x_cat = self.d_catconv(x_cat)

            x_cat = torch.cat([x_cat,x_fea2],dim=1)
            x_cat = self.d_sequnce_blocks[idx][1](x_cat)
            x_cat = self.d_catconv(x_cat)

            x_cat = torch.cat([x_cat,x_fea3],dim=1)
            x_cat = self.d_sequnce_blocks[idx][2](x_cat)
            x_cat = self.d_catconv(x_cat)
            x_cat = self.d_sequnce_blocks[idx][3](x_cat)
            x_d_fea = x_cat
            x_grad_branch = self.d_sequnce_blocks[idx][4:](x_cat)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1],x_grad_branch), 1)
            x = self.f_conv[idx](x)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)
            #print(sr.shape)
            results.append(sr)

            sr_out_grad = self.grad_tail[idx](x_grad_branch)
            sr_out_grad = self.add_mean(sr_out_grad )
            results_grad.append(sr_out_grad)
           
            
            if only_upsample:
                break

        return results,results_grad




if __name__ == '__main__':
    model = make_model(args)
    x = torch.randn(1,3,192,192)
    if args.add_gradient_branch:
        result,result_grad = model(x)
        for _ in result:
            print('result:',_.shape)

        for grad in result_grad:
            print('rrrrrr:',grad.shape)
    else:
        result = model(x)
        for _ in result:
            print('result:',_.shape)


