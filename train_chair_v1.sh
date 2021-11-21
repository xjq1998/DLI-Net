python train_v1.py \
    --ph_train_root /home/xjq/code/dataset/qmul-v1/chairs/photo \
    --sk_train_root /home/xjq/code/dataset/qmul-v1/chairs/sketch \
    --ph_test_root /home/xjq/code/dataset/qmul-v1/chairs/photo \
    --sk_test_root /home/xjq/code/dataset/qmul-v1/chairs/sketch \
    --ph_train_txt /home/xjq/code/dataset/qmul-v1/chairs/photo_train_name.txt \
    --ph_test_txt /home/xjq/code/dataset/qmul-v1/chairs/photo_test_name.txt \
    --sk_test_txt /home/xjq/code/dataset/qmul-v1/chairs/sketch_test_name.txt \
    --feature_type global \
    --lr 0.0001 \
    --batch_size 32 \
    --epoch 100 \
    --margin 0.1 \
    --k 0.5 \
    --norm_type 2norm \
    --result_dir /home/xjq/code/shoe/DLI-Net/results