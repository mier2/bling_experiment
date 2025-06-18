# Training

We utilize open-source framework  [LLaMA-Factory]() to conduct our training process.

Step 1: Please add the data path to the file_name field of ReasonFlux entry in [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json).

Step 2: Run command below  to train from a 32B model on 8 A100 GPUs. 
```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir train/LLaMA-Factory/data \
    --dataset ReasonFlux \
    --cutoff_len 2048 \
    --learning_rate 2e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-32B-Instruct/full \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_offload_config.json
```