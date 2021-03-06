CUDA_VISIBLE_DEVICES=1 \
python train_sketchy.py \
    --ph_train_root /home/xjq/code/dataset/sketchy/photo/tx_000100000000 \
    --sk_train_root /home/xjq/code/dataset/sketchy/sketch/tx_000100000000 \
    --ph_test_root /home/xjq/code/dataset/sketchy/photo/tx_000100000000 \
    --sk_test_root /home/xjq/code/dataset/sketchy/sketch/tx_000100000000 \
    --ph_train_txt /home/xjq/code/dataset/sketchy/photo_train_relative_path.txt \
    --ph_test_txt /home/xjq/code/dataset/sketchy/photo_test_relative_path.txt \
    --sk_test_txt /home/xjq/code/dataset/sketchy/sketch_test_relative_path.txt \
    --dict_path /home/xjq/code/dataset/sketchy/label_dict.npy \
    --feature_type global \
    --lr 0.00001 \
    --batch_size 32 \
    --epoch 300 \
    --margin 0.1 \
    --k 0.5 \
    --norm_type 2norm \
    --trip_weight 48 \
    --cls_weight 1 \
    --result_dir /home/xjq/code/shoe/DLI-Net/results
