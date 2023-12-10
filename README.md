# HMNN

首先进入到项目路径`HMNN`
```shell
cd HMNN
```
安装需要的依赖包
```shell
pip install torch
pip install tensorboard
```
在预训练阶段，我们使用了data/water.csv 和data/air2.csv.我们可以通过运行下面的命令执行下面第一阶段的预训练
```shell
python pretrain1.py --file_path "data/water.csv" --model_save_dir "water_model_result" --res_dir "water_res_result"
python pretrain1.py --file_path "data/air2.csv" --model_save_dir "air2_model_result" --res_dir "air2_res_result"
```
运行完之后你可以得到water 和 air 的预训练模型，默认存储在water_model_result和air2_model_result目录。模型的训练结果存储到water_res_result和air2_res_result目录。并且将选择出来的top2模型保存在water_res_result和air2_res_result目录。之后你可以利用预训练阶段一的模型，来完成预训练阶段二，只需运行一下代码：
```shell
python pretrain2.py --file_path "data/multisolvent2.csv" --model_save_dir "pretrain_model" --res_dir "pretrain_res" --water_model_save_path "water_model_result/waterdi6zhe6.ckpt" --air_model_save_path "air2_model_result/air2di9zhe6.ckpt" --water_select_expert_config "water_res_result/waterdi6zhe6_top2.csv" --air_select_expert_config "air2_res_result/air2di9zhe6_top2.csv"
```