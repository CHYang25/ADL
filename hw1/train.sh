    #! /bin/bash

    python multiple_choice.py \
    --train_file ${1} \
    --validation_file ${2}\
    --max_seq_length 128\
    --model_name_or_path roberta-base \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir ./ \
    --per_device_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --overwrite_output \
    --seed 4125252 \


    # Other unsused options:
    # --dataset_name
    # --dataset_config_name
    # --pad_to_max_length
    # --model_name_or_path
    # --config_name
    # --tokenizer_name
    # --use_slow_tokenizer
    # --weight_decay
    # --max_train_steps
    # --gradient_accumulation_steps
    # --lr_scheduler_type
    # --num_warmup_steps
    # --debug
    # --push_to_hub
    # --hub_model_id
    # --hub_token
    # --checkpointing_steps
    # --resume_from_checkpoint
