from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Cuda Is Available")

import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils import get_bnb_config
import random

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def model_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # model arguments
    parser.add_argument("--model_name_or_path", type=str, default="./model/Taiwan-LLM-7B-v2.0-chat")
    parser.add_argument("--trust_remote_code", type=bool, default=True)
    parser.add_argument("--cache_dir", type=str, default="./model/")


    args = parser.parse_known_args()[0]
    return args

def data_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # data arguments
    parser.add_argument("--validation_file", type=str, default="./data/public_test.json")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--few_shot_sample_num", type=int, default=3)

    args = parser.parse_known_args()[0]
    return args

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_accelerate_model(args, checkpoint_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=True,
        load_in_8bit=False,
        quantization_config=get_bnb_config(),
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=False,
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
        tokenizer_type="llama",
        trust_remote_code=args.trust_remote_code,
        use_auth_token=False,
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )


    print('Adding special tokens.')
    tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
    })

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    return model, tokenizer

model_args = model_parse_args()
data_args = data_parse_args()
args = argparse.Namespace(**vars(model_args), **vars(data_args))
print(args)

model, tokenizer = get_accelerate_model(args)

model.config.use_cache = False
print('loaded model')

def get_prompt_for_shots(instruction: str, prompt_option, **kwargs) -> str:
    '''Format the instruction as a prompt for LLM.'''
    if prompt_option == "zero":
        return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

    if prompt_option == "few":
        if "example" in kwargs:
            if isinstance(kwargs["example"], list):
                return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} \n範例：[{';'.join(kwargs['example'])}] ASSISTANT:"
            else:
                raise ValueError("Examples should be a list")
        else:
            raise ValueError("You should provide a few examples with few-shots")

    raise ValueError("You should choose prompt option from zero-shot(zero) and few-shots(few)")

def perplexity_for_shots(
    model, tokenizer, data, max_length=2048, prompt_option=None
):
    data_size = len(data)
    if prompt_option == "zero":
        data = data.add_column("example", [""]*data_size)
    elif prompt_option == "few":
        sorted_data = data.shuffle(seed=42)
        data = data.add_column("example", [[sorted_data[(i+j)%data_size]["instruction"] + ":" + sorted_data[(i+j)%data_size]["output"] + "\n" \
                                                for j in range(args.few_shot_sample_num)] for i in range(data_size)])
    else:
        raise ValueError("You should choose prompt option from zero-shot(zero) and few-shots(few)")

    instructions = [get_prompt_for_shots(x["instruction"], prompt_option, example=x["example"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size), position=0, leave=True):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids
        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

# evaluation dataset
eval_dataset = Dataset.from_json(path_or_paths=args.validation_file)

print("*** Zero-shot ***")
eval_result = perplexity_for_shots(model=model, tokenizer=tokenizer, data=eval_dataset , max_length=2048, prompt_option="zero")
print("***** eval metrics *****")
print("  num_example\t\t=\t{}".format(len(eval_dataset)))
print("  mean_perplexity\t=\t{}".format(eval_result["mean_perplexity"]))

print("*** Few-shots ***")
eval_result = perplexity_for_shots(model=model, tokenizer=tokenizer, data=eval_dataset, max_length=2048, prompt_option="few")
print("***** eval metrics *****")
print("  num_example\t\t=\t{}".format(len(eval_dataset)))
print("  mean_perplexity\t=\t{}".format(eval_result["mean_perplexity"]))