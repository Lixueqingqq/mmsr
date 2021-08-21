export CUDA_VISIBLE_DEVICES=0
nohup python -u main.py \
--data_dir="/home/datalab/ex_disk/work/AIDesignDataSets/SR_data"  \
--data_train="DFFlickr2K_Place" \
--data_test="DFFlickr2K_Place" \
--add_noise \
--add_unsupervised \
--add_gradient_branch \
--loss='1*primary_L1+0.1*dual_L1+1.5*perceptual_L1+0*tv_TV+0.05*gradientpixel_MSE+0.5*gradientbranch_L1' \
--scale=4 \
--patch_size=192 \
--Degenerate_type='bic' \
--model="DRN-S" \
--pre_train="./log/DRN-Sx4_bic_perceptual/model/model_best.pt" \
--test_every=1000 \
--epochs=1000 \
--batch_size=20 \
--lr=0.0001 \
--save="./log/DRN-Sx4_unsupervised_perceptual_bic_gradientbranch/" \
--save_results \
--print_every=20 \
> out_unsupervised_perceptual_bic_gradientbranch.log 2>&1 & 
#nohup python -u main.py \