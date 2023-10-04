#! /bin/bash

PYTHONPATH="/tmp2/b10902069/python_package:$(pwd)/include/"
export PYTHONPATH

case ${1} in
    "multiple_choice")
        CUDA_VISIBLE_DEVICES=1 python ./src/multiple_choice_train.py \
        --train_file /tmp2/b10902069/adl_hw1/train.json \
        --validation_file /tmp2/b10902069/adl_hw1/valid.json \
        --context_file /tmp2/b10902069/adl_hw1/context.json \
        --max_seq_length 128 \
        --model_name_or_path bert-base-chinese \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --output_dir /tmp2/b10902069/adl_hw1/multiple_choice_dir/ \
        --per_device_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --seed 4125252 \

        # Other unsused options:
        # --dataset_name
        # --dataset_config_name
        # --pad_to_max_length
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
        ;;
    "extractive")
        CUDA_VISIBLE_DEVICES=1 python ./src/extractive_train.py \
        --train_file /tmp2/b10902069/adl_hw1/train.json \
        --validation_file /tmp2/b10902069/adl_hw1/valid.json \
        --context_file /tmp2/b10902069/adl_hw1/context.json \
        --max_seq_length 384 \
        --model_name_or_path bert-base-chinese \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --output_dir /tmp2/b10902069/adl_hw1/extractive_dir/ \
        --per_device_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --seed 4125252 \

        # Other unsused options:
        # --dataset_name
        # --dataset_config_name
        # --preprocessing_num_workers
        # --do_predict
        # --test_file
        # --pad_to_max_length
        # --config_name
        # --tokenizer_name
        # --use_slow_tokenizer
        # --weight_decay
        # --max_train_steps
        # --gradient_accumulation_steps
        # --lr_scheduler_type
        # --num_warmup_steps
        # --doc_stride
        # --n_best_size
        # --null_score_diff_threshold
        # --version_2_with_negative
        # --max_answer_length
        # --max_train_samples
        # --max_eval_samples
        # --overwrite_cache
        # --max_predict_samples
        # --model_type
        # --push_to_hub
        # --hub_model_id
        # --hub_token
        # --trust_remote_code
        # --checkpointing_steps
        # --resume_from_checkpoint
        # --with_tracking
        # --report_to
        ;;
    *)
        echo "Please enter which model you want to train? (\"multiple_choice\"/\"extractive\")"
        ;;
esac


