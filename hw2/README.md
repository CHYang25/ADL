# ADL HW2 Fine-tuning mT5 - New Title Generation
Inspired by [This Link](https://github.com/KrishnanJothi/MT5_Language_identification_NLP/blob/main/MT5_fine-tuning.ipynb)
## Description
This is the second homework assignment for the ADL course at NTU in the fall of 2023. It involves fine-tuning a Multilingual Text-to-Text Transfer Transformer (mT5). The objective is to have the model generate an appropriate title for a given paragraph. We assess the performance using the Rouge score, which is a reliable metric for evaluating sequence model outputs.

## Directory Structure
This is the directory structure <span style="color:red">BEFORE RUNNING ```download.sh```</span>:
```
hw2
├── README.md
├── report.pdf
├── src
│   ├── mt5_condi_gen.py
│   ├── mt5_condi_gen_train.ipynb
│   └── mt5_condi_gen_train.py
├── download.sh
├── run.sh
└── train.sh
```
1. ```README.md```: Describe the details of this project.
2. ```report.pdf```: The report for hw2.
3. ```src```: The directory contains the source python scripts.
    - ```mt5_condi_gen.py```: The python script that generates the prediction result of the test dataset.
    - ```mt5_condi_gen_train.ipynb```: The python notebook that trains the ```MT5ForConditionalGeneration``` model and generates the result of the model's prediction. 
    - ```mt5_condi_gen_train.py```: The python script that trains the ```MT5ForConditionalGeneration``` model and generates the result of the model's prediction. It's basically the same as ```mt5_condi_gen_train.ipynb```.
4. ```download.sh```: Downloads the models, tokenizers and data
5. ```run.sh```: Run the ```./src/mt5_condi_gen.py``` script with options set and **generate the result**.
6. ```train.sh```: Run the ```./src/mt5_condi_gen_train.py``` script with options set and generate the result. Also, loss and rouge figure would be generated as well.

## Execution
(change directory to where the scripts are!!)
1. To download the dataset and the model
    ```
    ./download.sh
    ```
2. To generate the result
    ```
    ./run.sh ./data/public.jsonl ./data/output.jsonl
    ```
3. To reproduce the model and the result
    ```
    ./train.sh ./data/train.jsonl ./data/public.jsonl ./data/output.jsonl
    ```
    You can also use the GUI provided by Jupyter Notebook with ```mt5_condi_gen_train.ipynb```.
## Some Notes
some notes:
- We only train the decoder, encoder is already finished.
- No bonus is done considering my workload.
- Change twrouge.py line 11 to the following code segment if the ckiptagger can't download the zip file:
    ```
    if not os.path.exists(os.path.join(data_dir, "model_ws")):
        import zipfile
        import gdown
        
        file_id = "1NdLWUXDISBnwKDY3hhXxlHcmmt1Kpp68"
        url = f"https://drive.google.com/uc?id={file_id}"
        print(url)
        data_zip = os.path.join(download_dir, "data.zip")
        gdown.download(url, data_zip, quiet=False)
        
        with zipfile.ZipFile(data_zip, "r") as zip_ref:
            zip_ref.extractall(download_dir)
    ```
    The Error message:
    ```
    Access denied with the following error:

    Too many users have viewed or downloaded this file recently. Please
    try accessing the file again later. If the file you are trying to
    access is particularly large or is shared with many people, it may
    take up to 24 hours to be able to view or download the file. If you
    still can't access a file after 24 hours, contact your domain
    administrator. 

    You may still be able to access the file from the browser:

        https://drive.google.com/uc?id=1efHsY16pxK0lBD2gYCgCTnv1Swstq771 
    ```

## Useful Links
1. [Homework Description Slides](https://docs.google.com/presentation/d/1yJEQUtzFREeuEnkBTXei4SFEftnmnP3i05m0J5aGsg8/edit#slide=id.gd5d1d9d2d2_0_201)
2. About the accelerate package:
    - [accelerate](https://pypi.org/project/accelerate/)
    - [Basic tutorials](https://huggingface.co/docs/accelerate/basic_tutorials/migration)
3. About mT5 model:
    - [mt5 paper](https://arxiv.org/abs/2010.11934)
    - [mt5-small](https://huggingface.co/google/mt5-small)
    - [mt5 documentation](https://huggingface.co/docs/transformers/model_doc/mt5#mt5)
    - [t5 Tokenizer](https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/t5#transformers.T5Tokenizer)
    - [MT5ForConditionalGeneration source code](https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/mt5/modeling_mt5.py#L1546)
4. fine-tuning mt5:
    - [Metric - twrouge](https://github.com/moooooser999/ADL23-HW2)
    - [Deal with CUDA out of memory](https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-in-pytorch/#:~:text=Solution%20%231%3A%20Reduce%20Batch%20Size,and%20see%20if%20that%20helps.)
    - [About model.train() model.eval()](https://blog.csdn.net/weixin_44211968/article/details/123774649)
5.  Generation strategy: 
    - [About generation strategy](https://blog.csdn.net/bqw18744018044/article/details/126944119)
    - [generate](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)
    - [GenerationConfig](https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)