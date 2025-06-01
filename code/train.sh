train_dataset=$1
gpus=$2

epoch=2
learning_rate=1.0e-5

# 训练的权重的名字
ex_name="Qwen2___5-7B-Instruct-traindata_${train_dataset}"

# sft权重的保存路径
save_model=../models/${ex_name}/full/sft

# 训练
# export FORCE_TORCHRUN=1

CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli train \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ../data/ds_z2_config.json \
    --model_name_or_path Qwen/Qwen2___5-7B-Instruct \
    --dataset ${train_dataset} \
    --dataset_dir ../data \
    --template empty \
    --output_dir ${save_model} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epoch} \
    --plot_loss \
    --bf16 \
    --save_total_limit 5 \
    --temperature 0.7

