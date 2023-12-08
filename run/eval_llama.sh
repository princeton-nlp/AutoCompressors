#!/bin/bash -l
#SBATCH -J eval_llama
#SBATCH -N 1 -n 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:a100:1 --constraint gpu80
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 0-3:00:00




nvidia-smi
conda activate auto
base_model=${BASE:-"Llama-2-7b-hf"}
run_name=${NAME:-"ac_Llama-2-7b-hf_sub2_seg2_sum50_lr4e-4_bsz32_rand_accu/checkpoint-65000"}    # use llama-2-7b-hf for base model
block_size=${BLOCK:-8192}

total=${BATCH:-32}      # total batch size
bs=${SEQ:-2}            # batch size per device
lr=${LR:-8e-4}
warmup_steps=${WU:-5000}
save_steps=${SAVE:-5000}
segments_per_substep=${SEG:-2}
training_substeps=${SUB:-1}
summary_length=${SUM:-50}
summary_accumulation=${ACC:-true}
randomize_substeps=${RAND:-false}
num_train_epochs=1
segment_lengths=${SEGLEN:-"2048 2048"}
mode=${MODE:-CP}        #CP for compression, FA for full-attention
rope_theta=${ROPE:-10000}
segment_gradient_checkpointing=${CHECK:-false}

max_eval_samples=${MAXEVAL:-500}
num_nodes=${NUM_NODES:-1}
node=${NODE:-"localhost"}
################################

num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
total_per_device=$((${total}/${num_gpus}/${num_nodes}))
accu=$(( ${total_per_device} / ${bs} ))

run_name="checkpoints/${run_name}"
echo "Run: ${run_name}"

cache_dir=./.cache
out_dir=$run_name/eval_${block_size}

if [ $mode == FA ]; then
    export FA_EVAL=true
    out_dir=${out_dir}_fa
fi
mkdir -p $out_dir 

wandb disabled

export OMP_NUM_THREADS=8
header="torchrun \
--nnodes=$num_nodes \
--nproc_per_node=$num_gpus \
--rdzv-backend=c10d \
--rdzv-endpoint=$node:5546 \
train.py "


model_url="meta-llama/${base_model}"
arguments=(
    --report_to wandb
    --config_name $model_url
    --tokenizer_name $model_url
    --model_name_or_path $model_url
    --gradient_accumulation_steps $accu
    --per_device_eval_batch_size $bs
    --per_device_train_batch_size $bs
    --learning_rate $lr
    --warmup_steps $warmup_steps
    --do_eval
    --max_eval_samples $max_eval_samples
    --logging_steps 1
    --save_steps $save_steps
    --preprocessing_num_workers 6
    --dataloader_num_workers 6
    --cache_dir $cache_dir
    --add_special_tokens false
    --num_train_epochs ${num_train_epochs}
    --disable_tqdm true
    --resume_from_checkpoint true
    --log_level info
    --learning_rate $lr
    --output_dir $out_dir
    --use_fast_tokenizer false
    --summary_length $summary_length
    --accumulate_summary $summary_accumulation
    --remove_unused_columns false
    --segments_per_substep $segments_per_substep
    --training_substeps $training_substeps
    --randomize_substeps $randomize_substeps
    --segment_lengths $segment_lengths
    --segment_gradient_checkpointing $segment_gradient_checkpointing
    --bf16
    --run_name $run_name
    --rope_theta ${rope_theta}
    $@
)

echo "Evaluating on ${block_size} token sequences"
data="preprocessed_redpajama-weighted-disjoint_${block_size}"
arguments+=(--preprocessed_validation_datasets \
                    ${data}/arxiv \
                    ${data}/book \
                    ${data}/c4 \
                    ${data}/github \
                    ${data}/stack_exchange \
                    ${data}/wiki \
                    ${data}/cc/2019-30-head-en \
                    ${data}/cc/2019-30-middle-en \
                    ${data}/cc/2020-05-head-en \
                    ${data}/cc/2020-05-middle-en \
                    ${data}/cc/2021-04-head-en \
                    ${data}/cc/2021-04-middle-en \
                    ${data}/cc/2022-05-head-en \
                    ${data}/cc/2022-05-middle-en \
                    ${data}/cc/2023-06-head-en \
                    ${data}/cc/2023-06-middle-en \
                    )

if [[ $run_name == checkpoints/ac_Llama* ]]; then
    arguments+=(
    --lora
    --lora_path $run_name
    --lora_r 16
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target_modules q_proj v_proj o_proj k_proj
    --lora_modules_to_save embed_summary
    )
fi

#################

echo "Training ${base_model} with lr ${lr} on ${dataset}"
echo Outputting to $out_dir

echo command: echo "$header ${arguments[@]}"
$header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out
