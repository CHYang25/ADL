# ADL HW1 Fine-tuning Transformer

## Description
This assignment is the first homework for the ADL course at NTU in the fall of 2023. The project involves the implementation of two distinct models: the initial model focuses on paragraph selection, while the second one handles span selection. The process is provided with four Chinese paragraphs alongside a Chinese question. The first model's objective is to accurately identify the relevant paragraph based on the question. Following this, the second model is tasked with extracting the precise segment from the chosen paragraph. Further elaboration on this task will be provided in following discussions.

## Directory Structure
This is the directory structure <span style="color:red">BEFORE RUNNING ```download.sh```</span>:
```
hw1
├── README.md
├── download.sh
├── include
│   ├── __pycache__
│   └── utils_qa.py
├── report.pdf
├── run.sh
├── src
│   ├── __pycache__
│   ├── extractive_train.py
│   ├── main.py
│   └── multiple_choice_train.py
└── train.sh
```
1. ```README.md```: Describe the details of this project.
2. ```download.sh```: Downloads the models, tokenizers and data
3. ```include```: The directory contains included python scripts. ```utils_qa.py``` is imported by ```extractive_train.py```
4. ```report.pdf```: The report for hw1.
5. ```run.sh```: Run the ```src/main.py``` script with options set and generate the result.
6. ```src```: The directory contains the source python scripts.
    - ```extractive_train.py```: The python script that train the second model for extracting the precise segment from the chosen paragraph.
    - ```main.py```: The python script that import the two fine-tuned model and generate the result.
    - ```multiple_choice_train.py```: The python script that train the first model identifying the relevant paragraph based on the question.
7. ```train.sh```: Fine-tune the two models from two different pretrained models.

## Execution
### Steps of how to generate the result:
1. change directory to ```hw1```
2. run ```download.sh```: This would download both models to ```./model```
```
$ ./download.sh
```
3. run ```run.sh```: This would start the model and generate the result
```
$ ./run.sh <path_to_context.json> <path_to_test.json> <path_to_prediction.csv>
```
### Steps of how to reproduce the model:
Run ```train.sh``` to train both of the models.
```
# For mutliple choice model
$ ./train.sh multiple_choice <path_to_train.json> <path_to_valid.json> <path_to_context.json>

# For extractive model
$ ./train.sh extractive <path_to_train.json> <path_to_valid.json> <path_to_context.json>
```

## Useful Links
1. https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#autotokenizer
2. https://huggingface.co/transformers/v3.4.0/custom_datasets.html
3. https://huggingface.co/docs/transformers/tasks/multiple_choice
4. https://huggingface.co/docs/datasets/loading