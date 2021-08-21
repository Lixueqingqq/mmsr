import argparse
import utility
import numpy as np

parser = argparse.ArgumentParser(description='DRN')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--data_dir', type=str, default='data_path',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DF2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
# 48 * 4, 48 *8

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--add_noise', action='store_true',
                    help='do noise data augmentation')
parser.add_argument('--add_unsupervised', action='store_true',
                    help='add unsupervised data for training')

parser.add_argument('--Degenerate_type', type=str, default='bic',
                    help='the degenerate type of data for training,bic|bic&near')
parser.add_argument('--model', type=str, default='DRN-S',help='model name: DRN-S | DRN-L')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_blocks', type=int, default=30,
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--negval', type=float, default=0.2, 
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--up_type',type=str,default='pixelshuffle',
                    help='the type of upsample in drn.py')
parser.add_argument('--add_gradient_branch', action='store_true',
                    help='add gradient branch')

parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

parser.add_argument('--lr', type=float, default=1e-4, 
                    help='learning rate')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss', type=str, default='1*primary_L1+0.1*dual_L1+100*perceptual_L1+0.001*tv_TV+1*gradientpixel_MSE+0*gradientbranch_L1',
                    help='loss function configuration, L1|MSE')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
#parser.add_argument('--dual_weight', type=float, default=0.1,
#                    help='the weight of dual loss')
#parser.add_argument('--perloss_weight', type=float, default=10,
#                    help='the weight of perceptual loss')
#parser.add_argument('--tvloss_weight', type=float, default=0,
#                    help='the weight of tv loss')
#parser.add_argument('--gradientloss_weight', type=float, default=0,
#                    help='the weight of gradient loss')

parser.add_argument('--save', type=str, default='../experiment/',
                    help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

args = parser.parse_args()

utility.init_model(args)

# scale = [2,4] for 4x SR to load data
# scale = [2,4,8] for 8x SR to load data
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]
loss_compose = args.loss
for ll in args.loss.split('+'):
    weight,name_type = ll.split('*')
    if 'primary' in name_type:
        args.primary_weight = float(weight)
    if 'dual' in name_type:
        args.dual_weight = float(weight)
    if 'perceptual' in name_type:
        args.perloss_weight = float(weight)
    if 'tv' in name_type:
        args.tvloss_weight = float(weight)
    if 'gradientpixel' in name_type:
        args.gradientloss_weight = float(weight)
    if 'gradientbranch' in name_type:
        args.gradientbranch_weight = float(weight)


print('PPPPPPPPPPPPPPP:',args.add_unsupervised)

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

