#! /bin/bash

python3 ./src/mt5_condi_gen.py \
    --test_file ${1} \
    --result_file ${2} \
    --max_output_length=64 \
    --max_input_length=256 \
    --model_name_or_path ./model/ \
    --output_dir ./model/ \
    --seed 20231104 \