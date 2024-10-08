#! /bin/bash

PYTHONPATH="/tmp2/b10902069/python_package:$(pwd)/include/"
export PYTHONPATH

case ${1} in
    "multiple_choice")
        python3 ./src/multiple_choice_train.py \
        --train_file ${2} \
        --validation_file ${3} \
        --context_file ${4} \
        --max_seq_length 512 \
        --model_name_or_path bert-base-chinese \
        --learning_rate 3e-5 \
        --num_train_epochs 8 \
        --output_dir ./model/multiple_choice_dir/ \
        --per_device_eval_batch_size=4 \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=2 \
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
        python3 ./src/extractive_train.py \
        --train_file ${2} \
        --validation_file ${3} \
        --context_file ${4} \
        --max_seq_length 512 \
        --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --output_dir ./model/extractive_dir/ \
        --per_device_eval_batch_size=2 \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=1 \
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
        # --max_train_steps
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

    "combined")
        python3 ./src/combined_train.py \
        --train_file ${2} \
        --validation_file ${3}\
        --context_file ${4} \
        --max_seq_length 512 \
        --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --output_dir ./model/combined_dir/ \
        --per_device_eval_batch_size=2 \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=8 \
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
        # --max_train_steps
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
    "extractive_curve")
        python3 ./src/extractive_train_curve.py \
        --train_file ${2} \
        --validation_file ${3} \
        --context_file ${4} \
        --max_seq_length 512 \
        --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --output_dir ./model/extractive_curve_dir/ \
        --per_device_eval_batch_size=2 \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps=1 \
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
        # --max_train_steps
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
    "multiple_choice_scratch")
        python3 ./src/multiple_choice_train.py \
        --train_file ${2} \
        --validation_file ${3} \
        --context_file ${4} \
        --max_seq_length 512 \
        --model_name_or_path bert-base-chinese \
        --learning_rate 3e-5 \
        --num_train_epochs 8 \
        --output_dir ./model/multiple_choice_scratch_dir/ \
        --per_device_eval_batch_size=4 \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=2 \
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
        # --lr_scheduler_type
        # --num_warmup_steps
        # --debug
        # --push_to_hub
        # --hub_model_id
        # --hub_token
        # --checkpointing_steps
        # --resume_from_checkpoint
        ;;
    *)
        echo "==========================Please enter the options==========================" 
        echo "  For models fine-tuning: (multiple_choice/extractive)"
        echo "  For combined model fine-tuning: (combined)"
        echo "  To plot the curve: (extractive_curve)"
        echo "  To train the multiple_choice model from scratch: (multiple_choice_scratch)"
        ;;
esac