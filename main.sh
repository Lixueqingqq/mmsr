export CUDA_VISIBLE_DEVICES=2
python -u main.py \
--data_dir="/DataSets/super_resolution" \
--data_train="DFFlickr2K" \
--data_test="B100" \
--add_noise \
--scale=4 \
--patch_size=192 \
--model="DRN-S" \
--test_every=1000 \
--epochs=1000 \
--batch_size=24 \
--lr=0.0001 \
--save="./log/DRN-Sx4" \
--save_results \
--print_every=10