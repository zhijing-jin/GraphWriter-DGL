env CUDA_VISIBLE_DEVICES=1 python -u train.py --prop 2 --save_model tmp_model.pt --title

env CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 2 --save_model tmp_model.pt --title --train_file data/agenda/full_train.json   --valid_file data/agenda/full_valid.json   --test_file data/agenda/full_test.json


env CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 6 --save_model tmp_model1.pt --title > train_2.log 2>&1 &
env CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 6 --save_model tmp_model2.pt --title > train_3.log 2>&1 &
env CUDA_VISIBLE_DEVICES=0 python -u train.py --prop 6 --save_model tmp_model3.pt --title > train_4.log 2>&1 &
