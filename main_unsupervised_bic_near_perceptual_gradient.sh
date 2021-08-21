export CUDA_VISIBLE_DEVICES=1
nohup python -u main.py \
--data_dir="/home/datalab/ex_disk1/open_dataset/SR_data" \
--data_train="DFFlickr2K_Place" \
--data_test="DFFlickr2K_Place" \
--add_noise \
--add_unsupervised \
--loss='1*primary_L1+0.1*dual_L1+1.5*perceptual_L1+0*tv_TV+0.05*gradientpixel_MSE+0*gradientbranch_L1' \
--scale=4 \
--patch_size=192 \
--up_type='upconv' \
--Degenerate_type='bic&near' \
--model="DRN-S" \
--pre_train="./log/DRN-Sx4_unsupervised_bic-near/model/model_best.pt" \
--test_every=1000 \
--epochs=1000 \
--batch_size=22 \
--lr=0.0001 \
--save="./log/DRN-Sx4_unsupervised_perceptual_bic-near_gradient_upconv/" \
--save_results \
--print_every=20 \
> out_unsupervised_perceptual_bic_near_gradient_upconv.log 2>&1 & 
#nohup python -u main.py \