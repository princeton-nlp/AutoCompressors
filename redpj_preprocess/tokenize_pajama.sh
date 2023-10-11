#!/bin/bash -l
#SBATCH -J preprocess
#SBATCH -N 1 -n 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:a100:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH -t 1-0:00:00


block_size=${BLOCK:-6144}
train=${TRAIN:-1B}
validation=${VAL:-7M}
domain=${DOM}

conda activate dia

mkdir -p new_preprocessed_redpajama-weighted/${domain}

python -m preprocess \
--dataset redpajama-weighted/${domain} \
--tokenizer "meta-llama/Llama-2-7b-hf" \
--block_size ${block_size} \
--total_train ${train} \
--total_validation ${validation} \
--outdir new_preprocessed_redpajama-weighted/${domain} \





