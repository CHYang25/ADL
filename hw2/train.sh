#! /bin/bash

PYTHONPATH="/tmp2/b10902069/python_package:$(pwd)/include"
export PYTHONPATH

python3 ./src/train_decoder.py \
--train_file "/tmp2/b10902069/adl_hw2/train.jsonl" \
--validation_file "/tmp2/b10902069/adl_hw2/public.jsonl" \
--result_file "./result.jsonl" \
--max_seq_length 32 \
--model_name_or_path "google/mt5-small" \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--output_dir /tmp2/b10902069/adl_hw2/model \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \

# num_warmup_steps
# weight_decay