This is forked from [Qipeng Guo's repository](https://github.com/QipengGuo/GraphWriter-DGL/tree/e115d68e4a098a402acf4eecffdf99a7dbd2c62f), and excludes large data files.

# GraphWriter-DGL
In this example we implement the GraphWriter, [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/abs/1904.02342) in DGL. And the [author's code](https://github.com/rikdz/GraphWriter). 

## Dependencies
PyTorch >= 1.2  
tqdm   
pycoco 

## Usage
```
  sh run.sh
  sh test.sh
```

## Result on AGENDA
| |BLEU|METEOR| training time per epoch|
|-|-|-|-|
|paper|14.3+-1.01| 18.8+-0.28| 1970s|
|this repo|14.31+-0.34|19.74+-0.69| 1192s|

We use the author's code for the speed test, and our testbed is V100 GPU.

| |BLEU| detok BLEU| METEOR | 
|-|-|-|-|
|this repo, greedy, two layers| 13.97 +- 0.40| 13.78 +- 0.46| 18.76 +- 0.36|
|this repo, beam 4, length penalty 1.0, two layers| 14.66 +- 0.65| 14.53 +- 0.52| 19.50 +- 0.49|
|this repo, beam 4, length penalty 0.0, two layers| 14.33 +- 0.39| 14.09 +- 0.39| 18.63 +- 0.52|
|this repo, greedy, six layers| 14.17 +- 0.46| 14.01 +- 0.51| 19.18 +- 0.49|
|this repo, beam 4, length penalty 1.0, six layers| 14.31 +- 0.34| 14.35 +- 0.36| 19.74 +- 0.69|
|this repo, beam 4, length penalty 0.0, six layers| 14.40 +- 0.85| 14.15 +- 0.84| 18.86 +- 0.78|

We repeat the experiment five times. 
