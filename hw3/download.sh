#! /bin/bash

# peft config download
# For this assignment, we don't directly fine-tune the LLM, we tune the adaptation
# Thus, we only need to download the adaptation's configuration (peft config)

gdown https://drive.google.com/drive/folders/14GiRpPd1zvGgeHmw_AioVeXfqTonyUhr -O ./adapter_checkpoint --folder
