#! /bin/bash

python3 ./src/taiwan_llama_lora.py \
--model_name_or_path ${1} \
--adapter_path ${2} \
--test_file ${3} \
--result_file ${4} \