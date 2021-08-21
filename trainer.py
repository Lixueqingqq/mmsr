# -*- coding:utf-8 -*-
import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from loss import Get_gradient_nopadding 

class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.get_grad = Get_gradient_nopadding()
       
    
        #定义perceptual_loss feature extract
        if self.opt.perloss_weight>0:
            self.per_feature = self.model.perceptual_F
        # SR模型
        # 定义优化器
        self.optimizer = utility.make_optimizer(opt, self.model)
        # 定义学习率
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        # 对偶模型
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8
        
        self.start_time = time.time()
        
        self.writer = SummaryWriter(self.opt.save)

    def train(self):
        """
        训练
        """
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        self.writer.add_scalar('lr', lr, epoch)
        ploss_total = 0.0
        #ploss_primal = 0.0
        #ploss_dual = 0.0
        unsuper_num = 0
        for batch, (lr, hr, filename) in enumerate(self.loader_train):
            #print('this batch filename:',filename)
            for name in filename:
                if 'unsupervised' in name:
                    unsuper_num=unsuper_num+1
            

            lr, hr = self.prepare(lr, hr)
            timer_data.hold() # 统计数据读取时间
            timer_model.tic() # 统计模型时间
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            if self.opt.add_gradient_branch:
                sr,gd_sr = self.model(lr[0])
            else:
                sr = self.model(lr[0])  #假设是4x超分，会对lr[0]上采样到目标尺寸，然后下采样再上采样，最后输出结果是对应的lr[0].shape,lr[1].shape=lr[0].shape*2,hr.shape=lr[0].shape*4
            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            loss_primary = 0
            act_sample_num = sr[-1].shape[0]
            if "unsupervised" in ''.join(filename):
                #print(filename)
                for batch_subid, name in enumerate(filename):
                    if 'unsupervised' in name:
                        act_sample_num = act_sample_num - 1
                        hr[batch_subid,:,:,:] = sr[-1][batch_subid,:,:,:]
            
            element_num = act_sample_num*sr[-1].shape[1]*sr[-1].shape[2]*sr[-1].shape[3]
          
            loss_primary = self.loss('primary',sr[-1], hr,element_num)           

            for i in range(1, len(sr)):
                loss_primary += self.loss('primary', sr[i - 1 - len(sr)], lr[i - len(sr)])

            if self.opt.perloss_weight>0:
                loss_perceptual = self.loss('perceptual', self.per_feature(sr[-1],self.opt.rgb_range),self.per_feature(hr,self.opt.rgb_range),element_num)

            # compute dual loss
            loss_dual = self.loss('dual', sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss('dual', sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + loss_dual 
            if self.opt.perloss_weight>0:
                loss = loss + loss_perceptual

            if self.opt.tvloss_weight>0:
                loss_tv =  self.loss('tv', sr[-1])
                loss = loss + loss_tv

            # compute gradient pixel loss
            if self.opt.gradientloss_weight>0:
                loss_gradient = self.loss('gradientpixel',self.get_grad(sr[-1]),self.get_grad(hr))
                loss = loss + loss_gradient

            # compute gradient branch loss
            if self.opt.add_gradient_branch and self.opt.gradientbranch_weight>0:
                loss_gradient_branch = self.loss('gradientbranch',gd_sr[-1],self.get_grad(hr))
                for i in range(1, len(gd_sr)):
                    loss_gradient_branch += self.loss('gradientbranch', gd_sr[i - 1 - len(gd_sr)], self.get_grad(lr[i - len(gd_sr)]))
                loss = loss + loss_gradient_branch
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            ploss_total = ploss_total+loss.data.cpu().numpy()
            #ploss_primal = ploss_primal+loss_primary.data.cpu().numpy()
            #ploss_dual = ploss_dual+loss_dual.data.cpu().numpy()
           
            cur_iteration = batch + 1  + (epoch - 1) * self.opt.test_every
            max_iteration = self.opt.epochs * self.opt.test_every
            eta_seconds = ((time.time() - self.start_time) / cur_iteration) * (max_iteration - cur_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            visual_num = 0
            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\tModeTime:{:.1f}+DataTime:{:.1f}s\tEstTime:{}d'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(), eta_string))

                #self.writer.add_scalar('LossL1/train', loss, cur_iteration)
                self.writer.add_scalars('Loss/train', {'loss_primary':loss_primary}, cur_iteration)
                self.writer.add_scalars('Loss/train', {'loss_dual':loss_dual}, cur_iteration)
                if self.opt.perloss_weight>0:
                    self.writer.add_scalars('Loss/train', {'loss_perceptual':loss_perceptual}, cur_iteration)
                if self.opt.tvloss_weight>0:
                    self.writer.add_scalars('Loss/train', {'loss_tv':loss_tv}, cur_iteration)
                if self.opt.gradientloss_weight>0:
                    self.writer.add_scalars('Loss/train', {'loss_gradientpixel':loss_gradient}, cur_iteration)
                if self.opt.add_gradient_branch and self.opt.gradientbranch_weight>0:
                    self.writer.add_scalars('Loss/train', {'loss_gradienbranch':loss_gradient_branch}, cur_iteration)

                #self.writer.add_scalar('LossL1/train', ploss_total/(batch+1), cur_iteration)
                #self.writer.add_scalars('Loss/train', {'loss_primary':ploss_primal/(batch+1)}, cur_iteration)
                #self.writer.add_scalars('Loss/train', {'loss_dual':ploss_dual/(batch+1)}, cur_iteration)
                #if (batch + 1) //self.opt.print_every == 2:
                #    break   

                while visual_num<min(sr[-1].shape[0],10):
                    normalized = sr[-1][visual_num].clone().data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                    ndarr = normalized.byte().cpu().numpy()
                    #print('nnnnnn:',ndarr.shape)

                    normalized_lr = lr[0][visual_num].clone().data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                    ndarr_lr = normalized_lr.byte().cpu().numpy()
                    #print('nnnnnnlrrrrr:',ndarr_lr.shape)
                    lr_data = np.zeros(ndarr.shape)
                    lr_data[:,:ndarr_lr.shape[1],:ndarr_lr.shape[2]]=ndarr_lr
                    ndarr = np.concatenate((ndarr,lr_data),axis=2)
                    if 'unsupervised' in filename[visual_num]:
                        combine = ndarr
                    else:
                        normalized_hr = hr[visual_num].clone().data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                        ndarr_hr = normalized_hr.byte().cpu().numpy()
                        combine = np.concatenate((ndarr_hr, ndarr), axis=2)
                        if self.opt.add_gradient_branch and self.opt.gradientbranch_weight>0:
                            hr_grad = self.get_grad(hr[visual_num].unsqueeze(0))
                            hr_grad = hr_grad.squeeze(0).data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                            hr_grad = hr_grad.byte().cpu().numpy()
                            sr_gradbranch = gd_sr[-1][visual_num].clone().data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                            sr_gradbranch = sr_gradbranch.byte().cpu().numpy()
                            combine = np.concatenate((hr_grad,sr_gradbranch,combine), axis=2)
                    self.writer.add_image('image_%d/train'%(visual_num),np.uint8(combine),cur_iteration)
                    visual_num=visual_num+1

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.writer.add_scalar('LossL1/train', ploss_total/len(self.loader_train), epoch)

        print('unsuper_num is:',unsuper_num,unsuper_num/30000)
       
        # 调整学习率
        self.step()

    def test(self):
        """
        验证
        """
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()
        visual_num = 0

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                eval_ssim = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                eval_num = 0
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = ('unsupervised' in filename) #and (hr.nelement() == 1)
                    if not no_eval:
                        #print('llll filename:',filename)
                        lr, hr = self.prepare(lr, hr)
                        eval_num = eval_num + 1
                    else:
                        #print('uuuuuuuuuu:',filename)
                        lr, = self.prepare(lr)
                    
                    if self.opt.add_gradient_branch:
                        sr,gd_sr = self.model(lr[0])
                    else:
                        sr = self.model(lr[0])
        
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                        eval_ssim += utility.calc_ssim(sr, hr, s, self.opt.rgb_range)

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)
                    if random.random()<0.5:
                        if visual_num<3:
                            normalized = sr.clone()[0].data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                            ndarr = normalized.byte().cpu().numpy()

                            normalized_lr = lr[0].clone()[0].data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                            ndarr_lr = normalized_lr.byte().cpu().numpy()
                            lr_data = np.zeros(ndarr.shape)
                            lr_data[:,:ndarr_lr.shape[1],:ndarr_lr.shape[2]]=ndarr_lr
                            ndarr = np.concatenate((ndarr,lr_data),axis=2)
                            #print('ndarr:',ndarr.shape)
                            if 'unsupervised' in filename:
                                combine = ndarr
                            else:
                                normalized_hr = hr.clone()[0].data.mul(255 / self.opt.rgb_range).clamp(0, 255)
                                ndarr_hr = normalized_hr.byte().cpu().numpy()
                                combine = np.concatenate((ndarr_hr, ndarr), axis=2)
                                #print('combiner:',combine.shape)
                            self.writer.add_image('image_%d/valid'%(visual_num),np.uint8(combine),epoch)
                            visual_num=visual_num+1

                    if _>60:
                        break
                self.ckp.log[-1, si] = eval_psnr / eval_num #len(self.loader_test)
                self.writer.add_scalar('test_psnr_x'+str(s), self.ckp.log[-1, si], epoch)
                self.writer.add_scalar('test_ssim_x'+str(s), eval_ssim/eval_num, epoch)
                
            
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
