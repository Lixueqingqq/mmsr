export CUDA_VISIBLE_DEVICES=1
nohup python -u main.py \
--data_dir="/home/datalab/ex_disk/work/AIDesignDataSets/SR_data" \
--data_train="DFFlickr2K_Place" \
--data_test="DFFlickr2K_Place" \
--add_noise \
--loss='1*primary_L1+0.1*dual_L1+1*perceptual_L1+0*tv_TV+0*gradientpixel_MSE+0*gradientbranch_L1' \
--scale=4 \
--patch_size=192 \
--Degenerate_type='bic' \
--model="DRN-S" \
--pre_train="./log/DRN-Sx4_bic_perceptual/model/model_best.pt" \
--test_every=1000 \
--epochs=1000 \
--batch_size=26 \
--lr=0.0001 \
--save="./log/DRN-Sx4_bic_perceptual/" \
--save_results \
--print_every=20 \
> out_bic_perceptual.log 2>&1 & 
#nohup python -u main.py \