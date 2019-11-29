env CUDA_VISIBLE_DEVICES=0 python -u train.py --save_model tmp_xx.pt --test  --title --lp 1.0 --beam_size 4
perl eval/detokenizer.perl -l en < tmp_gold.txt > tmp_gold.txt.a
perl eval/detokenizer.perl -l en < tmp_pred.txt > tmp_pred.txt.a
perl eval/multi-bleu-detok.perl tmp_gold.txt.a < tmp_pred.txt.a
