# first include all the headers
from parse_arg import parse_args
import json
import logging
import math
import os
import random

import datasets
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    get_scheduler, # Not sure which to chose
    get_linear_schedule_with_warmup,
)
from tw_rouge.tw_rouge.twrouge import get_rouge

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def main():
    logger = get_logger(__name__)
    
    args = parse_args()

    # acclerator setup
    accelerator_log_kwargs = {}
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    torch.cuda.caching_allocator_alloc(int(8e6), device=1)

    # parsing data
    train_df = pd.read_json(args.train_file, lines=True)
    valid_df = pd.read_json(args.validation_file, lines=True)

    logger.info("=====DATA CLEANING=====")
    train_df = train_df.drop_duplicates(subset='title', keep=False).drop_duplicates(subset='maintext', keep=False)
    valid_df = valid_df.drop_duplicates(subset='title', keep=False).drop_duplicates(subset='maintext', keep=False)
    train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    eval_batch_size = args.per_device_eval_batch_size * args.gradient_accumulation_steps

    # Adding prefix text to the input, which helps the model to understand the fine-tuning task objective
    train_df["maintext"] = '<idf.lang>' + train_df["maintext"] 
    valid_df["main"] = '<idf.lang>' + valid_df["maintext"]
    logger.info(f"=====CLEANED TRAIN DATA===== | Shape:{train_df.shape}")
    print(train_df.head())
    logger.info(f"=====CLEANED VALID DATA===== | Shape:{valid_df.shape}")
    print(valid_df.head())

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
    logger.info(f"The device is using: {device}")
    model.to(device)

    LANG_TOKEN_MAPPING = {'identify language': '<idf.lang>'}
    special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
    print(len(tokenizer))
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.all_special_tokens)
    print(len(tokenizer), model.config.vocab_size)
    # model.resize_token_embeddings(len(tokenizer)) 
    # FIXME

    encode_main_text = lambda text: tokenizer.encode(text=text, 
                                                        return_tensors = 'pt',
                                                        padding = 'max_length',
                                                        truncation = True,
                                                        max_length = args.max_seq_length,
                                                    )[0]
    def encode_feature(feature):
        # encode an entire feature
        input_text = feature["maintext"]
        output_text = feature["title"]

        if input_text == None or output_text == None: return None

        input_tokens = encode_main_text(input_text)
        output_tokens = encode_main_text(output_text)

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

    total_train_steps = math.ceil(train_df.shape[0] / args.per_device_train_batch_size) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=args.num_warmup_steps*args.gradient_accumulation_steps,
                        num_training_steps=total_train_steps*args.gradient_accumulation_steps,
                    )

    tokenizer, model, optimizer, lr_scheduler = accelerator.prepare(
        tokenizer, model, optimizer, lr_scheduler)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_df)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = \
                    {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    loss_plt_list = []

    for epoch in range(args.num_train_epochs):
        train_dataset = encode_dataset(train_df, train_batch_size)
        
        for idx, (input_batch, label_batch) in enumerate(train_dataset):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model.forward(input_ids=input_batch, labels=label_batch)
                loss = outputs.loss
                loss_plt_list.append(loss.item())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                
        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            # show the information
            print(f"Epoch: {epoch} | Step: {idx} | Train loss: {loss.item()} | lr: {lr_scheduler.get_last_lr()[0]}")

if __name__ == "__main__":
    main()