CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 6 --save_model tmp_useless.pt
CUDA_VISIBLE_DEVICES=1 python -u train.py --prop 6 --save_model tmp_useless.pt
 --title > tmp_normal.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.py --prop 6 --save_model tmp_useless.pt --save_dataset data_w_s.pickle --train_file data/webnlg_sent/train.json  --valid_file data/webnlg_sent/valid.json  --test_file data/webnlg_sent/test.json


