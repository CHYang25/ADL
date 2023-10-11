#! /bin/bash

PYTHONPATH="$(pwd)/include/"
export PYTHONPATH

python3 ./src/main.py \
--test_file ${2} \
--context_file ${1} \
--result_file ${3} \
--max_seq_length 512 \
--model_name_or_path_multiple_choice ./model/multiple_choice_dir/ \
--model_name_or_path_extractive ./model/extractive_dir/ \
--output_dir ./model/ \

# Other unsused options:
# --with_tracking
# --gradient_accumulation_steps
# --pad_to_max_length
# --preprocessing_num_workers
# --overwrite_cache
# --per_device_eval_batch_size
# --version_2_with_negative
# --max_answer_length
# --null_score_diff_threshold
# --doc_stride
# --n_best_size