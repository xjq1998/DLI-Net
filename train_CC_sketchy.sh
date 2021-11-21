python train_sketchy.py \
    --ph_train_root /home/xjq/code/dataset/sketchy/photo/tx_000100000000 \
    --sk_train_root /home/xjq/code/dataset/sketchy/sketch/tx_000100000000 \
    --ph_test_root /home/xjq/code/dataset/sketchy/photo/tx_000100000000 \
    --sk_test_root /home/xjq/code/dataset/sketchy/sketch/tx_000100000000 \
    --ph_train_txt /home/xjq/code/dataset/sketchy/photo_seen_train.txt \
    --ph_test_txt /home/xjq/code/dataset/sketchy/photo_seen_test.txt \
    --sk_test_txt /home/xjq/code/dataset/sketchy/sketch_seen_test.txt \
    --dict_path /home/xjq/code/dataset/sketchy/zs_label_dict.npy \
    --feature_type mid \
    --lr 0.00001 \
    --batch_size 32 \
    --epoch 20 \
    --margin 0.1 \
    --self_interaction \
    --cross_interaction \
    --k 0.5 \
    --trip_weight 48.0 \
    --cls_weight 1.0 \
    --norm_type 2norm \
    --result_dir /home/xjq/code/shoe/DLI-Net/results