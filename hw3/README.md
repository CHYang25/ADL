# ADL HW3 Instruction-tuning LLaMA2 - Classical Chinese Translation
Inspired by: [Link](https://github.com/artidoro/qlora/blob/main/qlora.py)

## Description
This is the third assignment for the ADL course at NTU - fall 2023. It involves instruction-tuning a Taiwan LlaMa Model. The objective is to translate plain Chinese into classical Chinese, or to translate the classical Chinese into plain Chinese. We assess the preformance using perplexity metirc.


## Directory Structure
This is the directory structure <span style="color:red">AFTER RUNNING ```download.sh```</span>:
```
.
├── adapter_checkpoint
├── download.sh
├── image
│   └── loss.png
├── prediction.json
├── README.md
├── report.pdf
├── run.sh
└── src
    ├── chinese_llama_lora_train.ipynb
    ├── chinese_llama_lora_train.py
    ├── ppl.py
    ├── taiwan_llama_lora.py
    ├── taiwan_llama_lora_train.ipynb
    ├── taiwan_llama_lora_train_loss.ipynb
    ├── taiwan_llama_lora_train_loss.py
    ├── taiwan_llama_lora_train.py
    ├── taiwan_llama_shot_inference.ipynb
    ├── taiwan_llama_shot_inference.py
    └── utils.py
```
1. ```README.md```: Describe the details of this project.
2. ```report.pdf```: The report for hw3.
3. ```src```: The directory contains the source python scripts.
4. ```download.sh```: Downloads the adapter checkpoint.
5. ```run.sh```: Run the ```./src/taiwan_llama_lora.py``` script with options set and **generate the result**.
6. ```adapter_checkpoint```:  The adapter checkpoint that should be loaded in order to do inference.
7. ```image```: The directory that contains a loss png file.
8. ```prediction.json```: The prediction file by inferencing the model on the private data set.
9. Note: 
    - In the source code, the Taiwan-Llama Model is declared in a direcotry ```./model/```. Make sure you set the argument ```model_name_or_path``` to the one you use.
    - The ```ppl.py``` is under the ```src``` directory. Make sure you use it correctly: ```python3 ./src/ppl.py [options]```
## Environment Requirement
1. accelerate==0.24.1
2. transformers==4.34.1
3. bitsandbytes==0.41.1
4. peft==0.6.0
5. datasets==2.5.2
6. evaluate==0.4.0
7. sentencepiece==0.1.99

## Execution
<span style="color:red">**The following code should exectued under the root directory of this package.**</span>

1. To download the adapter checkpoint:
    ```
    ./download.sh
    ```

2. To generate the prediction file by inferencing a fine-tuned model:
    ```
    ./run.sh <path_to_llama_model> ./adapter_checkpoint <path_to_input_json> <path_to_output_json>
    ```

3. To reproduce the adapter model alongside Taiwan-Llama (you may want to change the arguments on your own by options, or modify the default value in the source code):
    ```
    python3 ./src/taiwan_llama_lora_train.py [options]
    ```

4. To reproduce the result of different inference strategies:
    ```
    python3 ./src/taiwan_llama_shot_inference.py [options]
    ```

5. To reproduce the learning curve on the public dataset:
    ```
    python3 ./src/taiwan_llama_lora_train_loss.py [options]
    ```

6. To reproduce the adapter model alongside Chinese-Llama (you may want to change the arguments on your own by options, or modify the default value in the source code):
    ```
    python3 ./src/chinese_llama_lora_train.py [options]
    ```

## Useful Links
1. Majorly inspired by: [Link](https://github.com/artidoro/qlora/blob/main/qlora.py)
2. Homework Description: [Link](https://docs.google.com/presentation/d/1bZyF83pI9WZq558QDNsO9E7vl2B6PQJLLpu4V5EBo9A/edit#slide=id.g297f132dcc7_4_6)
3. LLaMA2: 
    - [Huggingface API Documentation](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/llama2)
    - [Official Website](https://ai.meta.com/llama/)
    - [Source Code of Huggingface API](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
4. peft: 
    - [Module Description](https://pypi.org/project/peft/)
    - [Source code of prepare_model_for_kbit_training](https://github.com/huggingface/peft/blob/f1ecfa6ae6eba599ae89decbf47b339d8c9d39a3/src/peft/utils/other.py#L67)
    - [Huggingface Example](https://huggingface.co/docs/transformers/v4.35.2/en/peft#train-a-peft-adapter)
    - [Huggingface API Documentation](https://huggingface.co/docs/peft/package_reference/peft_model)
5. LoRA (Low Rank Adaptation):
    - [LoRA Description](https://huggingface.co/docs/peft/conceptual_guides/lora)
    - [LoRA Paper](https://arxiv.org/pdf/2106.09685.pdf)
    - [QLoRA Description](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
    - [Source code of LoRA Model generate()](https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L1053)
    - [Source code of LoRA Model](https://github.com/huggingface/peft/blob/f1ecfa6ae6eba599ae89decbf47b339d8c9d39a3/src/peft/tuners/lora/model.py)
6. perplexity: 
    - [Perplexity Explanatory Video](https://www.youtube.com/watch?v=NURcDHhYe98)
    - [Perplexity Description](https://huggingface.co/docs/transformers/perplexity)S
7. Trainer: 
    - [Huggingface Description](https://huggingface.co/docs/transformers/main_classes/trainer)
    - [Source code of Trainer](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/trainer.py#L231)
    - [Huggingface API Documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer)
    - [Loss history of Trainer](https://discuss.huggingface.co/t/how-to-get-the-loss-history-when-use-trainer-train/17486)_
8. Other Tools:
    - [torch.Tensor.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)
    - [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html)
    - [Datasets](https://huggingface.co/docs/datasets/v1.2.0/processing.html)
    - [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)
    - [Gradient Checkpointing](https://github.com/cybertronai/gradient-checkpointing)
    - [HfArgumentParser](https://huggingface.co/transformers/v4.2.2/_modules/transformers/hf_argparser.html)
9. Chinese-LLaMa: 
    - [Model Github Repo](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
    - [Model Description](https://huggingface.co/hfl/chinese-llama-2-7b)