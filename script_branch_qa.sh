gpu_i=0
dataset_name=msqa
model_name=/data1/PTLM/t5_base/
#model_name=t5-large
use_context=True
max_len=2048
seed=0
lr=3e-4
if [ $model_name == t5_large ]; then
  ga=12
elif [ $model_name == t5_3b ]; then
  ga=24
else
  ga=8
fi

python run_one2branch_qa.py --dataset_name $dataset_name --model_name $model_name --ga $ga --gpu $gpu_i \
  --epoch_num 20 --seed $seed --acc_epoch -1 --lr $lr --max_len $max_len --use_context $use_context