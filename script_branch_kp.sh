gpu_i=0
dataset_name=kp20k_small
#dataset_name=kptimes_small
#dataset_name=stackex_small
model_name=t5-base
#model_name=t5-large
ga=16

seed=0

python run_one2branch_kp.py --dataset_name $dataset_name --model_name $model_name --ga $ga \
--gpu $gpu_i --epoch_num 10 --acc_epoch 0 --min_beam_num 8 --max_beam_num 60 --seed $seed --use_negative False

python run_one2branch_kp.py --dataset_name $dataset_name --model_name $model_name --ga $ga \
--gpu $gpu_i --only_eval_train True --min_beam_num 8 --max_beam_num 60 --seed $seed
#
python run_one2branch_kp.py --dataset_name $dataset_name --model_name $model_name --ga $ga \
--gpu $gpu_i --use_negative True --epoch_num 5 --min_beam_num 8 --max_beam_num 60 --seed $seed