nvidia-smi

# You can override the default parameters by passing variables to the script
base_model=${BASE:-"opt-2.7b"}
total=${BATCH:-16}      # total batch size
bs=${SEQ:-1}            # batch size per device
lr=${LR:-2e-5}
warmup_steps=${WU:-1000}
save_steps=${SAVE:-1000}
num_gpus=${NUM_GPUS:-1}
segments_per_substep=${SEG:-2}
training_substeps=${SUB:-2}
summary_length=${SUM:-50}
summary_accumulation=${ACC:-true}
randomize_substeps=${RAND:-true}
segment_gradient_checkpointing=${CHECK:-false}
num_train_epochs=1

train_domains=(Books3 Github FreeLaw Wikipedia)
eval_domains=(Books3 Github FreeLaw Wikipedia Gutenberg HackerNews ArXiv YoutubeSubtitles)

################################

total_per_device=$((${total}/${num_gpus}))
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
header="torchrun --standalone \
--nnodes=1 \
--nproc_per_node=$num_gpus \
train.py "

model_url="facebook/${base_model}"

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
    --do_eval
    --evaluation_strategy steps
    --logging_steps 1
    --eval_steps $save_steps
    --save_steps $save_steps
    --preprocessing_num_workers 10
    --dataloader_num_workers 10
    --cache_dir $cache_dir
    --add_special_tokens false
    --max_eval_samples 2000
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
    $@
)

echo "Training on 6K token sequences"
arguments+=(--preprocessed_train_datasets)
for train_domain in ${train_domains[@]}; do
    arguments+=("awettig/Pile-${train_domain}-0.5B-6K-opt");
done
arguments+=(--preprocessed_validation_datasets)
for valid_domain in ${eval_domains[@]}; do
    arguments+=("awettig/Pile-${valid_domain}-0.5B-6K-opt");
done

#################

echo "Training ${base_model} with lr ${lr} on ${dataset}"
echo train domains: ${train_domains[@]}
echo valid domains: ${eval_domains[@]}
echo Outputting to $out_dir

echo command: echo "$header ${arguments[@]}"
$header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out
