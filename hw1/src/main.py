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
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=1, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
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
        tokenizer_e = AutoTokenizer.from_pretrained(
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

    def preprocess_function_m(examples):
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
            preprocess_function_m, batched=True, remove_columns=raw_datasets["test"].column_names
        )

    test_dataset = processed_datasets["test"]
    data_collator = DataCollatorForMultipleChoice(
            tokenizer_m, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=1)

    model_m, test_dataloader = accelerator.prepare(model_m, test_dataloader)
    model_m.eval()

    # multiple_choice model output
    logger.info("***** Running Multiple Choice Model *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    first_model_output = []

    progress_bar = tqdm(range(len(test_dataset)), disable=not accelerator.is_local_main_process)
    paragraph_list = [[i.as_py() for i in segment] for segment in raw_datasets.data["test"]["paragraphs"]]
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model_m(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        first_model_output.append(paragraph_list[step][predictions.item()])
        progress_bar.update(1)
        
    # Second Model -- Extractive
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer_e.padding_side == "right"

    if args.max_seq_length > tokenizer_e.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer_e.model_max_length}). Using max_seq_length={tokenizer_e.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer_e.model_max_length)

    question_column_name = "question"
    context_column_name = "relevant"
    # Pre-processing
    def preprocess_function_e(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer_e(
            examples[question_column_name],
            [context[i] for i in examples[context_column_name]],
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        ) if pad_on_right else tokenizer_e(
            [context[i] for i in examples[context_column_name]],
            examples[question_column_name],
            truncation="only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_examples = raw_datasets["test"]
    eval_examples = eval_examples.remove_columns("paragraphs")
    eval_examples = eval_examples.add_column(name="relevant", column=first_model_output)

    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            preprocess_function_e,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_examples.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )

    data_collator = DataCollatorWithPadding(tokenizer_e, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Post-processing:
    def post_processing_function_e(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            context_file=context, # the context.json is converted into a list here, called context. In utils_qa.py, it's called context_file
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        
        # >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
        return formatted_predictions

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    model_e, eval_dataloader = accelerator.prepare(model_e, eval_dataloader)
    model_e.eval()

    logger.info("***** Running Extractive Model *****")
    logger.info(f"  Num examples = {len(test_dataset)}")

    progress_bar_2 = tqdm(range(len(test_dataset)), disable=not accelerator.is_local_main_process)

    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model_e(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
            progress_bar_2.update(1)

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function_e(eval_examples, eval_dataset, outputs_numpy)

    logger.info("***** Saving the result *****")
    progress_bar_3 = tqdm(range(len(test_dataset)), disable=not accelerator.is_local_main_process)  
    with open("result.csv", "w") as f:
        f.write("id,answer\n")
        for pair in prediction:
            id_, text_ = pair["id"], pair["prediction_text"]
            f.write(f"{id_},\"{text_}\"\n")
            progress_bar_3.update(1)

if __name__ == "__main__":
    main()