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

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils import get_prompt, get_bnb_config
from ppl import perplexity

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def model_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # model arguments
    parser.add_argument("--model_name_or_path", type=str, default="./model/Taiwan-LLM-7B-v2.0-chat")
    parser.add_argument("--trust_remote_code", type=bool, default=True)
    parser.add_argument("--adapter_path", type=str, default="./adapter_checkpoint")
    parser.add_argument("--load_from_checkpoint_for_eval", type=bool, default=True)

    args = parser.parse_known_args()[0]
    return args

def data_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # data arguments
    parser.add_argument("--test_file", type=str, default="./data/private_test.json")
    parser.add_argument("--result_file", type=str, default="./prediction.json")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)

    args = parser.parse_known_args()[0]
    return args

class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: str = field(
        default="./model/"
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="fp4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    def __init__(self, cache_dir:str, double_quant:bool, quant_type:str, lora_r:int, lora_alpha:float, lora_dropout:float, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

def train_parse_args():
    args = TrainingArguments(
            cache_dir="./model/",
            double_quant=True,
            quant_type="fp4",
            lora_r=64,
            lora_alpha=16,
            lora_dropout=0.0,

            output_dir="./model/",
            optim="paged_adamw_32bit",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=250,
            weight_decay=0.0,
            learning_rate=1e-4,
            remove_unused_columns=False,
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            do_train=True,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            logging_steps=10, # FIXME10
            group_by_length=True,
            save_strategy="steps",
            save_steps=25, #FIXME 250
            save_total_limit=50,
    )
    return args

def gen_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # Generation strategy
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--use_cache", type=bool, default=True)

    # Hyperparameters for logit manipulation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--diversity_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

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

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # modules = find_all_linear_names(args, model)
    # print(modules)
    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    return model, tokenizer

model_args = model_parse_args()
data_args = data_parse_args()
train_args = train_parse_args()
gen_args = gen_parse_args()
train_args.generation_config = transformers.GenerationConfig(**vars(gen_args))
args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(train_args), **vars(gen_args))
print(args)

model, tokenizer = get_accelerate_model(args, args.adapter_path if args.load_from_checkpoint_for_eval else None)

model.config.use_cache = False
print('loaded model')
set_seed(args.seed)

test_dataset = Dataset.from_json(path_or_paths=args.test_file)

print("*** Test ***")
print("  num_example = {}".format(len(test_dataset)))

model.eval()
test_result_lst = []
progress_bar = tqdm(range(len(test_dataset)), position=0, leave=True)
for idx, example in enumerate(test_dataset):
    input_ids = torch.tensor([tokenizer.bos_token_id] + \
                    tokenizer.encode(text=get_prompt(example['instruction']), add_special_tokens=False, max_length=args.max_target_len)).unsqueeze(0)
    prediction_ids = model.generate(input_ids=input_ids, generation_config=train_args.generation_config)
    prediction = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    test_result_lst.append({"id":example["id"], "output": prediction})
    progress_bar.update(1)

# Verifying the result
print("***** Prediction Samples *****")
print(test_result_lst[:5])

for x in test_result_lst:
    if "ASSISTANT" in x["output"]:
        x["output"] = x["output"].split("ASSISTANT: ")[-1]
    if "USER" in x["output"]:
        x["output"] = x["output"].split("USER: ")[-1]
    if "答案" in x["output"]:
        x["output"] = x["output"].split("答案：")[-1]
    if "：" in x["output"]:
        x["output"] = x["output"].split("：")[-1]
    if "。" in x["output"]:
        x["output"] = x["output"].split("。")[-1]
    if "你是" in x["output"]:
        x["output"] = x["output"].split("你是")[0]
    if "你要" in x["output"]:
        x["output"] = x["output"].split("你要")[0]

for x in test_result_lst:
    print(x)

with open(args.result_file, "w") as f:
    json.dump(test_result_lst, f, indent=4)