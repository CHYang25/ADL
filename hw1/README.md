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
│   ├── combined_train.py
│   ├── extractive_train_curve.py
│   ├── extractive_train.py
│   ├── main.py
│   ├── multiple_choice_train.py
│   ├── multiple_choice_train_scratch.py
│   └── __pycache__
└── train.sh
```
1. ```README.md```: Describe the details of this project.
2. ```download.sh```: Downloads the models, tokenizers and data
3. ```include```: The directory contains included python scripts. ```utils_qa.py``` is imported by ```extractive_train.py```, ```extractive_train_curve.py``` and ```combined_train.py```.
4. ```report.pdf```: The report for hw1.
5. ```run.sh```: Run the ```./src/main.py``` script with options set and **generate the result**.
6. ```src```: The directory contains the source python scripts.
    - ```combined_train.py```: The python script that train the merged model for directly extracting the precise segment from all the four paragraphs. This is for the bonus part.
    - ```extractive_train_curve.py```: The python script that train the second model for extracting the precise segment from the chosen paragraph. It would also plot the learning curve of loss and the exact match on **validation dataset.**
    - ```extractive_train.py```: The python script that train the second model for extracting the precise segment from the chosen paragraph.
    - ```main.py```: The python script that import the two fine-tuned model and generate the result.
    - ```multiple_choice_train.py```: The python script that train the first model identifying the relevant paragraph based on the question.
    - ```multiple_choice_train_scratch.py```: The python script that train the first model from scratch. In other words, the pre-trained model's weights are not loaded.
7. ```train.sh``` would start the following python scripts: 
    - Fine-tune the two models from two different pre-trained models: ```extractive_train.py``` and ```multiple_choice_train.py```
    - Fine-tune a merged model from the extractive pre-trained model: ```combined_train.py```
    - Fine-tune the extractive model and plot the learning curves: ```extractive_train_curve.py```
    - Train the multiple-choice model from scratch: ```multiple_choice_train_scratch.py```

## Execution
### Steps to generate the result:
1. change directory to ```hw1```
2. run ```download.sh```: This would download both models to ```./model/```, and the datasets to ```./data/```
```
$ ./download.sh
```
3. run ```run.sh```: This would start the model and generate the result
```
$ ./run.sh ./data/context.json ./data/test.json ./prediction.csv
```

### Steps to reproduce the model:
Run ```train.sh``` to train both of the models.
```
# For mutliple choice model
$ ./train.sh multiple_choice ./data/train.json ./data/valid.json ./data/context.json

# For extractive model
$ ./train.sh extractive ./data/train.json ./data/valid.json ./data/context.json
```
Then you'll have two models ready for ```main.py```. (or you can download them, as we mentioned from the last part)

### Steps to fine-tune the combined model(bonus):
Run ```train.sh``` to train the model:
```
$ ./train.sh combined ./data/train.json ./data/valid.json ./data/context.json
```

### Steps to plot the curves:
Run ```train.sh``` to train the model and plot the curves:
```
$ ./train.sh extractive_curve ./data/train.json ./data/valid.json ./data/context.json
```

### Steps to train the multiple_choice model from scratch:
Run ```train.sh``` to train the model:
```
$ ./train.sh multiple_choice_scratch ./data/train.json ./data/valid.json ./data/context.json
```

## Useful Links
1. https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#autotokenizer
2. https://huggingface.co/transformers/v3.4.0/custom_datasets.html
3. https://huggingface.co/docs/transformers/tasks/multiple_choice
4. https://huggingface.co/docs/datasets/loading