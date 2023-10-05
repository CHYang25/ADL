import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import send_example_telemetry

# for multiple_choice model
from transformers import AutoModelForMultipleChoice, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
# for extractive model
from transformers import AutoModelForQuestionAnswering, DataCollatorWithPadding, EvalPrediction
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Run and generate the result of the two models")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A json file containing the testing data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None, help="A json file containing the context of data."
    )
    parser.add_argument(
        "--model_name_or_path_multiple_choice",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path_extractive",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final result.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    args = parser.parse_args()

    return args

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # label_name = "label" if "label" in features[0].keys() else "labels"
        # labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        # batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def main():
    args = parse_args()

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

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

    # Load Models
    if args.model_name_or_path_multiple_choice and args.model_name_or_path_extractive:
        config_m = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_multiple_choice,
            cache_dir=args.output_dir,
        )
        tokenizer_m = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_multiple_choice,
            use_fast=True,
            cache_dir=args.output_dir,
        )
        model_m = AutoModelForMultipleChoice.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_multiple_choice,
            cache_dir=args.output_dir,
        )

        config_e = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_extractive,
            cache_dir=args.output_dir
        )
        tokenizer_e = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_extractive,
            use_fast=True,
            cache_dir=args.output_dir
        )
        model_e = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path_extractive,
            cache_dir=args.output_dir,
        )
    else:
        raise Exception("You should specify the two fine-tuned models")

    # setups
    device = accelerator.device
    logger.info(f"The device is using: {device}")
    model_m.to(device)
    model_e.to(device)

    # First Model -- Multiple Choice
    # Process Data
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={"test":args.test_file})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model_m.get_input_embeddings().weight.shape[0]
    if len(tokenizer_m) > embedding_size:
        model_m.resize_token_embeddings(len(tokenizer))

    context_name = "paragraphs"
    with open(args.context_file, "r") as f:
        context = json.loads(f.read())
    question_name = "question"
    label_name = "relevant"

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[question] * 4 for question in examples[question_name]]
        second_sentences = [[context[i] for i in context_ids] for context_ids in examples[context_name]]

        # Flatten out
        first_sentences = list(chain(*first_sentences)) # the paragraph options
        second_sentences = list(chain(*second_sentences)) # the question

        # Tokenize
        tokenized_examples = tokenizer_m(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
        )

    test_dataset = processed_datasets["test"]
    data_collator = DataCollatorForMultipleChoice(
            tokenizer_m, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=1)

    model_m, test_dataloader = accelerator.prepare(model_m, test_dataloader)
    model_m.eval()

    # multiple_choice model output
    first_model_output = []

    paragraph_list = [[i.as_py() for i in segment] for segment in raw_datasets.data["test"]["paragraphs"]]
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model_m(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        first_model_output.append(context[paragraph_list[step][predictions.item()]])
        
    # Second Model -- Extractive
    
    model_e,  = accelerator.prepare(model_e, )
    model_e.eval()


if __name__ == "__main__":
    main()