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

import matplotlib.pyplot as plt

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def model_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # model arguments
    parser.add_argument("--model_name_or_path", type=str, default="./model/Taiwan-LLM-7B-v2.0-chat")
    parser.add_argument("--trust_remote_code", type=bool, default=True)

    args = parser.parse_known_args()[0]
    return args

def data_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama Model With Adaptor by Instruction-Tuning")

    # data arguments
    parser.add_argument("--train_file", type=str, default="./data/public_test.json")
    parser.add_argument("--validation_file", type=str, default="./data/public_test.json")
    parser.add_argument("--test_file", type=str, default="./data/private_test.json")
    parser.add_argument("--result_file", type=str, default="./prediction.json")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)

    args = parser.parse_known_args()[0]
    if args.train_file is None or args.validation_file is None or args.result_file is None:
        raise ValueError("Need train file, validation file, and the result file specification")
    else:
        # neither do train_file nor validation file is None, so
        extension = args.train_file.split(".")[-1] # to see what extension the file is
        print(extension)
        assert extension == "json", "train_file should be a json file"
        extension = args.validation_file.split(".")[-1] # to see what extension the file is
        assert extension == "json", "validation_file should be a json file"
        extension = args.test_file.split(".")[-1] # to see what extension the file is
        assert extension == "json", "test_file should be a json file"
        extension = args.result_file.split(".")[-1] # to see what extension the file is
        assert extension == "json", "result_file should be a json file"
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
            logging_steps=5, # FIXME10
            group_by_length=True,
            save_strategy="steps",
            save_steps=250, #FIXME 250
            save_total_limit=50,
    )
    return args

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        kwargs["model"].save_pretrained(args.output_dir)

        pytorch_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

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

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" if "output" in example.keys() else "" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def load_dataset(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    # dataset = dataset.map()
    train_dataset = Dataset.from_json(path_or_paths=args.train_file)
    eval_dataset = Dataset.from_json(path_or_paths=args.validation_file)
    test_dataset = Dataset.from_json(path_or_paths=args.test_file)

    train_dataset = train_dataset.map(lambda x: {
        'input': get_prompt(x['instruction']),
        'output': x['output']
    })

    if args.group_by_length:
        train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.max_source_len,
        target_max_len=args.max_target_len,
        predict_with_generate=False,
        train_on_source=False,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator
    )

"""## Training"""

model_args = model_parse_args()
data_args = data_parse_args()
train_args = train_parse_args()
args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(train_args))
print(args)

model, tokenizer = get_accelerate_model(args)

model.config.use_cache = False
print('loaded model')
set_seed(args.seed)

data_module = load_dataset(tokenizer=tokenizer, args=args)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    **{k:v for k,v in data_module.items() if k != 'test_dataset'},
)

## call_backs
trainer.add_callback(SavePeftModelCallback)

# Verifying the datatypes and parameter counts before training.
model.print_trainable_parameters()
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items(): total+= v
for k, v in dtypes.items():
    print(k, v, v/total)

# All metric
all_metrics = {}

print("*** Train ***")
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

all_metrics.update(metrics)
loss_plt_list = [example["loss"] for example in trainer.state.log_history[:-1]] # the last one has no "loss" key

plt.plot(loss_plt_list, label="loss")
plt.legend()
plt.show()