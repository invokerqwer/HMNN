# HMNN

Project path:`HMNN`
```shell
cd HMNN
```
Dependence packages：
```shell
pip install torch
pip install tensorboard
```
In the pretraining process, we utilized the "data/water.csv" and "data/air2.csv" datasets. To perform the first stage of pretraining, execute the following commands:
```shell
python pretrain1.py --file_path "data/water.csv" --model_save_dir "water_model_result" --res_dir "water_res_result"
python pretrain1.py --file_path "data/air2.csv" --model_save_dir "air2_model_result" --res_dir "air2_res_result"
```
After running the commands, you will obtain pretraining models for Water and Air dataset, which are by default stored in the "water_model_result" and "air2_model_result" directories. The training results of the models are saved in the "water_res_result" and "air2_res_result" directories. Additionally, the top 2 selected models are saved in the "water_res_result" and "air2_res_result" directories.

Subsequently, you can use the models from the first pretraining stage to complete the second pretraining stage. Simply run the following code:

```shell
python pretrain2.py --file_path "data/multisolvent2.csv" --model_save_dir "pretrain_model" --res_dir "pretrain_res" --water_model_save_path "water_model_result/waterdi6zhe6.ckpt" --air_model_save_path "air2_model_result/air2di9zhe6.ckpt" --water_select_expert_config "water_res_result/waterdi6zhe6_top2.csv" --air_select_expert_config "air2_res_result/air2di9zhe6_top2.csv"
```
