#! /bin/bash

PYTHONPATH="/tmp2/b10902069/python_package:$(pwd)/include/"
export PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python3 ./src/main.py \
--test_file /tmp2/b10902069/adl_hw1/test.json \
--context_file /tmp2/b10902069/adl_hw1/context.json \
--max_seq_length 128 \
--model_name_or_path_multiple_choice /tmp2/b10902069/adl_hw1/multiple_choice_dir/ \
--model_name_or_path_extractive /tmp2/b10902069/adl_hw1/extractive_dir/ \
--output_dir /tmp2/b10902069/adl_hw1 \

# Other unsused options:
# --with_tracking
# --gradient_accumulation_steps
# --pad_to_max_length