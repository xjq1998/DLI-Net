CUDA_VISIBLE_DEVICES=1 \
python train_v2.py \
    --ph_train_root /home/xjq/code/dataset/qmul_v2/ShoeV2/trainB \
    --sk_train_root /home/xjq/code/dataset/qmul_v2/ShoeV2/trainA \
    --ph_test_root /home/xjq/code/dataset/qmul_v2/ShoeV2/testB \
    --sk_test_root /home/xjq/code/dataset/qmul_v2/ShoeV2/testA \
    --ph_train_txt /home/xjq/code/dataset/qmul_v2/ShoeV2/photo_train.txt \
    --ph_test_txt /home/xjq/code/dataset/qmul_v2/ShoeV2/photo_test.txt \
    --sk_test_txt /home/xjq/code/dataset/qmul_v2/ShoeV2/sketch_test.txt \
    --feature_type mid \
    --lr 0.0001 \
    --batch_size 32 \
    --epoch 100 \
    --margin 0.1 \
    --self_interaction \
    --cross_interaction \
    --k 0.5 \
    --norm_type 2norm \
    --result_dir /home/xjq/code/shoe/DLI-Net/results