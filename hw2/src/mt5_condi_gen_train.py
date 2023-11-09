"""
Title Generation with mT5 Fine-tuning Script

Description:
This Python script is designed for fine-tuning a multilingual T5 (mT5) model on a Title Generation task. 
It imports essential libraries, defines command-line arguments for customization, and checks file extensions. 
The script is intended to be run from the command line, allowing users to specify parameters for the training process.

Usage:
- Place your training data in a JSONL file and set the file path using the `--train_file` argument.
- Specify the validation data in a separate JSONL file and provide the file path with `--validation_file`.
- Define the desired output file path using `--result_file` where the generated titles will be saved.
- Adjust hyperparameters like learning rate, batch sizes, etc., using the provided arguments.

Note:
This script assumes access to a CUDA-enabled GPU for GPU acceleration, as it checks for GPU availability.

Example Command:
$ python3 mt5_condi_gen_train.py \
    --train_file=/content/data/train.jsonl \
    --validation_file=/content/data/public.jsonl \
    --result_file=/content/result.jsonl \
    --model_name_or_path=google/mt5-small \
    --learning_rate=1e-3 \
    --num_train_epochs=6 \
    --output_dir=/content/model/ \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --seed=20231104 \
    --weight_decay=0.0
"""

import torch
# check if there's any cuda device available
print(torch.cuda.is_available())

import json
import math

from accelerate import Accelerator
from accelerate.utils import set_seed

import transformers
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
from tw_rouge import get_rouge

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Finetune a mT5 model on Title Generation")
    parser.add_argument("--train_file", type=str, default="/content/data/train.jsonl")
    parser.add_argument("--validation_file", type=str, default="/content/data/public.jsonl")
    parser.add_argument("--result_file", type=str, default="/content/result.jsonl")
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--model_name_or_path", type=str, default="google/mt5-small")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="/content/model/")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20231104)
    parser.add_argument("--weight_decay", type=float, default=0.0)


    args = parser.parse_args()
    if args.train_file is None or args.validation_file is None or args.result_file is None:
        raise ValueError("Need train file, validation file, and the result file specification")
    else:
        # neither do train_file nor validation file is None, so
        extension = args.train_file.split(".")[-1] # to see what extension the file is
        print(extension)
        assert extension == "jsonl", "train_file should be a jsonl file"
        extension = args.validation_file.split(".")[-1] # to see what extension the file is
        assert extension == "jsonl", "validation_file should be a jsonl file"
        extension = args.result_file.split(".")[-1] # to see what extension the file is
        assert extension == "jsonl", "result_file should be a jsonl file"
    return args

def main():
    args = parse_args()

    # acclerator setup
    accelerator_log_kwargs = {}
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # parsing data
    train_df = pd.read_json(args.train_file, lines=True)
    eval_df = pd.read_json(args.validation_file, lines=True)

    print("=====DATA CLEANING=====")
    train_df = train_df.drop_duplicates(subset='title', keep=False).drop_duplicates(subset='maintext', keep=False)
    eval_df = eval_df.drop_duplicates(subset='title', keep=False).drop_duplicates(subset='maintext', keep=False)

    dropout_symbol = [" ", "\n"]
    for s in dropout_symbol:
        train_df["maintext"] = train_df["maintext"].str.replace(s, "")
        eval_df["maintext"] = eval_df["maintext"].str.replace(s, "")

    print(max([len(s) for s in train_df["maintext"]]))

    train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    eval_batch_size = args.per_device_eval_batch_size * args.gradient_accumulation_steps

    # Adding prefix text to the input, which helps the model to understand the fine-tuning task objective
    train_df["maintext"] = '<idf.lang>' + train_df["maintext"]
    eval_df["maintext"] = '<idf.lang>' + eval_df["maintext"]
    print(f"=====CLEANED TRAIN DATA===== | Shape:{train_df.shape}")
    print(train_df.head())
    print(f"=====CLEANED VALID DATA===== | Shape:{eval_df.shape}")
    print(eval_df.head())

    # Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                use_fast=True,
                trust_remote_code=True,
                cache_dir=args.output_dir
            )
    model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            trust_remote_code=True,
            cache_dir=args.output_dir,
        )

    device = accelerator.device
    print(f"The device is using: {device}")
    model.to(device)

    LANG_TOKEN_MAPPING = {'identify language': '<idf.lang>'}
    special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
    print(len(tokenizer))
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.all_special_tokens)
    print(len(tokenizer), model.config.vocab_size)
    model.resize_token_embeddings(len(tokenizer))

    encode_main_text = lambda text, length: tokenizer.encode(text=text,
                                                    return_tensors = 'pt',
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = length,
                                                )[0]
    def encode_feature(feature):
        # encode an entire feature
        input_text = feature["maintext"]
        output_text = feature["title"]

        if input_text == None or output_text == None: return None

        # note that the input and output should have different length constraint
        # or both inputs and outputs would have encoding too long, causing out of memory
        input_tokens = encode_main_text(input_text, args.max_input_length)
        output_tokens = encode_main_text(output_text, args.max_output_length)

        return input_tokens, output_tokens

    def encode_batch(batch):
        # encode an entire batch
        inputs = []
        outputs = []

        for index, feature in batch.iterrows():
            formatted_data = encode_feature(feature)
            if formatted_data is None: continue

            input_tokens, output_tokens = formatted_data
            inputs.append(input_tokens.unsqueeze(0))
            outputs.append(output_tokens.unsqueeze(0))

        batch_input_tokens = torch.cat(inputs).cuda()
        batch_output_ids = torch.cat(outputs).cuda()

        return batch_input_tokens, batch_output_ids

    def encode_dataset(dataset, batch_size):
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(dataset), batch_size):
            raw_batch = dataset[i:i+batch_size]
            yield encode_batch(raw_batch)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    total_train_steps = math.ceil(train_df.shape[0] / (args.per_device_train_batch_size*args.gradient_accumulation_steps)) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_train_steps/2)
    tokenizer, model, optimizer, lr_scheduler = accelerator.prepare(tokenizer, model, optimizer, lr_scheduler)

    # Training
    progress_bar = tqdm(range(total_train_steps), disable=not accelerator.is_local_main_process)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_df)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    train_gen_config = GenerationConfig(max_length=args.max_output_length,
                                        num_beams=10,
                                        length_penalty=1,
                                        no_repeat_ngram_size=2,
                                        early_stopping=True)
    train_gen = lambda input_tokens: model.generate(input_tokens, generation_config=train_gen_config)

    loss_plt_list = []
    rouge_plt_list = {"rouge-1":[], "rouge-2":[], "rouge-l":[]}

    min_lost = 100
    for epoch in range(args.num_train_epochs):
        train_dataset = encode_dataset(train_df, train_batch_size)

        for idx, (input_batch, label_batch) in enumerate(train_dataset):
            with accelerator.accumulate(model):
                outputs = model.forward(input_ids=input_batch, labels=label_batch)
                loss = outputs.loss

                accelerator.backward(loss)

                loss_plt_list.append(loss.item())
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            progress_bar.update(1)

            # Evaluate the rouge score every 10 steps
            if idx % 10 == 0:
                pdt_lst = []
                lb_lst = []
                output_tokens = train_gen(input_batch)
                for t, l in zip(output_tokens, label_batch):
                    pdt_lst.append(tokenizer.decode(t, skip_special_tokens=True))
                    lb_lst.append(tokenizer.decode(l, skip_special_tokens=True))
                rouge = get_rouge(pdt_lst, lb_lst)
                rouge_plt_list["rouge-1"].append(rouge["rouge-1"]["f"])
                rouge_plt_list["rouge-2"].append(rouge["rouge-2"]["f"])
                rouge_plt_list["rouge-l"].append(rouge["rouge-l"]["f"])

                print("Rouge-1:{} | Rouge-2:{} | Rouge-l:{}".format(rouge["rouge-1"]["f"], rouge["rouge-2"]["f"], rouge["rouge-l"]["f"]))
                del pdt_lst, lb_lst

            # show the information
            if idx % 10 == 0:
                print(f"Epoch: {epoch} | Step: {idx} | Train loss: {loss.item()} | lr: {lr_scheduler.get_last_lr()[0]} | Min loss:{min_lost}")

            if (idx > 1000 or epoch > 0) and loss.item() < min_lost:
                min_lost = loss.item()
                print(f"Epoch: {epoch} | Step: {idx} | Train loss: {loss.item()} | lr: {lr_scheduler.get_last_lr()[0]} | Min loss:{min_lost}")

                # Save the model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    model.eval()

    # pure greedy
    greedy_config = GenerationConfig(max_length=args.max_output_length)
    greedy = lambda input_tokens: model.generate(input_tokens, generation_config=greedy_config)
    # pure beam search
    beam_config = GenerationConfig(max_length=args.max_output_length,
                                num_beams=10,
                                length_penalty=1,
                                no_repeat_ngram_size=2,
                                early_stopping=True)
    beam = lambda input_tokens: model.generate(input_tokens, generation_config=beam_config)
    # sampling with temperature
    sampling_temp_config = GenerationConfig(max_length=args.max_output_length,
                                            do_sample=True,
                                            top_k=0,
                                            temperature=0.7)
    sampling_temp = lambda input_tokens: model.generate(input_tokens,generation_config=sampling_temp_config)
    # topK_sampling
    topK_sampling_config = GenerationConfig(max_length=args.max_output_length,
                                            do_sample=True,
                                            top_k=model.config.vocab_size//5)
    topK_sampling = lambda input_tokens: model.generate(input_tokens, generation_config=topK_sampling_config)
    # topP with topK sampling
    topP_sampling_config = GenerationConfig(max_length=args.max_output_length,
                                            do_sample=True,
                                            top_p=0.95,
                                            top_k=model.config.vocab_size//5)
    topP_sampling = lambda input_tokens: model.generate(input_tokens, generation_config=topP_sampling_config)

    generation_strategy = {"greedy":greedy, "beam":beam, "sampling_temp":sampling_temp, "topK":topK_sampling, "topP":topP_sampling}

    best_name = None
    best_score = 0
    for gen_name, gen in generation_strategy.items():
        print(f"using generation strategy : {gen_name}")
        progress_bar = tqdm(range(100), disable=not accelerator.is_local_main_process)
        pdt_lst, lb_lst = [], []
        for idx, feature in enumerate(eval_df.iloc):
            input_tokens = encode_main_text(feature["maintext"], args.max_input_length)
            input_tokens = input_tokens.unsqueeze(0).cuda()

            with torch.no_grad():
                outputs = gen(input_tokens)
                for output_tokens in outputs:
                    prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)
                    if len(prediction) == 0:
                        prediction = " "
                    pdt_lst.append(prediction)
                    lb_lst.append(feature["title"])
            progress_bar.update(1)

            if idx == 100: break
            # We first compare the result of first 100 set of datas
            # And chose the one with the highest rouge_score sum and choose it to generate the rest of the datas

        print(gen_name, ":")
        final_rouge = get_rouge(pdt_lst, lb_lst)
        score = [final_rouge["rouge-1"]["f"]*100, final_rouge["rouge-2"]["f"]*100, final_rouge["rouge-l"]["f"]*100]
        print(f"rouge-1: {score[0]}")
        print(f"rouge-2: {score[1]}")
        print(f"rouge-l: {score[2]}\n")

        if best_score < score[0] + score [1] + score[2]:
            best_score = score[0] + score[1] + score[2];
            best_name = gen_name

        del pdt_lst, lb_lst

    print(f"using generation strategy : {best_name}")
    total_eval_steps = math.ceil(eval_df.shape[0])
    progress_bar = tqdm(range(total_eval_steps), disable=not accelerator.is_local_main_process)

    pdt_lst, lb_lst = [], []
    for idx, feature in enumerate(eval_df.iloc):
        input_tokens = encode_main_text(feature["maintext"], args.max_input_length)
        input_tokens = input_tokens.unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = generation_strategy[best_name](input_tokens)
            for output_tokens in outputs:
                prediction = tokenizer.decode(output_tokens, skip_special_tokens=True)
                pdt_lst.append(prediction)
                lb_lst.append(feature["title"])
        progress_bar.update(1)

    final_rouge = get_rouge(pdt_lst, lb_lst)
    score = [final_rouge["rouge-1"]["f"]*100, final_rouge["rouge-2"]["f"]*100, final_rouge["rouge-l"]["f"]*100]
    print(f"rouge-1: {score[0]}")
    print(f"rouge-2: {score[1]}")
    print(f"rouge-l: {score[2]}")

    plt.plot(loss_plt_list, label="loss")
    plt.legend()
    plt.savefig("./figure/loss.png")
    plt.plot(rouge_plt_list["rouge-1"], label="rouge-1")
    plt.plot(rouge_plt_list["rouge-2"], label="rouge-2")
    plt.plot(rouge_plt_list["rouge-l"], label="rouge-l")
    plt.legend()
    plt.savefig("./figure/rouge.png")

    result_dict = [{"title": title_, "id": str(feature_["id"])} for feature_, title_ in zip(eval_df.iloc, pdt_lst)]

    with open(args.result_file, "w") as f:
        for result_pair in result_dict:
            json.dump(result_pair, f)
            f.write("\n")

if __name__ == "__main__":
    main()