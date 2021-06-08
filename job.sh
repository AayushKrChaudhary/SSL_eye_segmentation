expname_set=$"After_review_ETRA_a100"
batch_size=$"6"
mode_set=$"ssl"
ssl_value=$"5"
labeltype_value=$"all"

python3 train_SSL.py --model densenet --deviceID 0 --epochs 250 --expname $expname_set --mode $mode_set --labeltype $labeltype_value --SSLvalue $ssl_value --dataset "Semantic_Segmentation_Dataset"> ./logs/{$expname_set}_train.txt

