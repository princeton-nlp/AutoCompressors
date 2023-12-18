nvidia-smi

# You can override the default parameters by passing variables to the script
base_model=${BASE:-"Llama-2-7b-hf"}
total=${BATCH:-32}      # total batch size
bs=${SEQ:-2}            # batch size per device
lr=${LR:-8e-4}
warmup_steps=${WU:-5000}
save_steps=${SAVE:-5000}
num_gpus=${NUM_GPUS:-1}
segments_per_substep=${SEG:-2}
training_substeps=${SUB:-2}
summary_length=${SUM:-50}
num_nodes=${NUM_NODES:-1}
node=${NODE:-"localhost"}
summary_accumulation=${ACC:-true}
randomize_substeps=${RAND:-true}
segment_gradient_checkpointing=${CHECK:-false}
num_train_epochs=1

train_domains=(Books3 Github FreeLaw Wikipedia)
eval_domains=(Books3 Github FreeLaw Wikipedia Gutenberg HackerNews ArXiv YoutubeSubtitles)

################################

total_per_device=$((${total}/${num_gpus}/${num_nodes}))
accu=$(( ${total_per_device} / ${bs} ))

run_name_suffix="sub${training_substeps}_seg${segments_per_substep}_sum${summary_length}_lr${lr}_bsz${total}"
if [[ ${randomize_substeps} == true ]]; then
    run_name_suffix+="_rand"
fi
if [[ $summary_accumulation == true ]]; then
    run_name_suffix+="_accu"
fi
if [[ ${segment_gradient_checkpointing} == true ]]; then
    run_name+="_check"
fi
run_name="ac_${base_model}_${run_name_suffix}"

echo "Run: ${run_name}"

cache_dir=./.cache
out_dir=checkpoints/$run_name
mkdir -p $out_dir

export WANDB_DIR=$out_dir

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
    --do_train
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
    --run_name $run_name
    --output_dir $out_dir
    --summary_length $summary_length
    --accumulate_summary $summary_accumulation
    --remove_unused_columns false
    --segments_per_substep $segments_per_substep
    --training_substeps $training_substeps
    --randomize_substeps $randomize_substeps
    --segment_gradient_checkpointing $segment_gradient_checkpointing
    --bf16
    --lora
    --lora_r 16
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target_modules q_proj v_proj o_proj k_proj
    --use_fast_tokenizer false
    --lora_modules_to_save embed_summary
    $@
)

echo "Training on 6K token sequences"
arguments+=(--preprocessed_train_datasets awettig/RedPajama-combined-15B-6K-llama)

#echo "Training on 8K token sequences"
#arguments+=(--preprocessed_train_datasets awettig/RedPajama-combined-15B-8K-llama)

#################

echo "Training ${base_model} with lr ${lr} on ${dataset}"
echo train domains: ${train_domains[@]}
echo valid domains: ${eval_domains[@]}
echo Outputting to $out_dir

echo command: echo "$header ${arguments[@]}"
$header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out
