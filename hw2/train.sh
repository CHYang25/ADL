#! /bin/bash

python3 ./src/mt5_condi_gen_train.py \
   --train_file ${1} \
   --validation_file ${2} \
   --result_file ${3} \
   --max_output_length=64 \
   --max_input_length=256 \
   --model_name_or_path google/mt5-small \
   --learning_rate=1e-3 \
   --num_train_epochs=6 \
   --output_dir ./model/ \
   --per_device_train_batch_size=1 \
   --per_device_eval_batch_size=1 \
   --gradient_accumulation_steps=16 \
   --seed 20231104 \
   --weight_decay=0 \