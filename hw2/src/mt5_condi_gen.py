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
)
from tw_rouge import get_rouge

from tqdm.auto import tqdm
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Finetune a mT5 model on Title Generation")
    parser.add_argument("--test_file", type=str, default="/content/data/public.jsonl")
    parser.add_argument("--result_file", type=str, default="/content/result.jsonl")
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--model_name_or_path", type=str, default="google/mt5-small")
    parser.add_argument("--output_dir", type=str, default="/content/model/")
    parser.add_argument("--seed", type=int, default=20231104)

    args = parser.parse_args()
    if args.test_file is None or args.result_file is None:
        raise ValueError("Need test file, and the result file specification")
    else:
        # neither do train_file nor test file is None, so
        extension = args.test_file.split(".")[-1] # to see what extension the file is
        print(extension)
        assert extension == "jsonl", "test_file should be a jsonl file"
        extension = args.result_file.split(".")[-1] # to see what extension the file is
        assert extension == "jsonl", "result_file should be a jsonl file"
    return args

def main():
    args = parse_args()

    # acclerator setup
    accelerator_log_kwargs = {}
    accelerator = Accelerator(**accelerator_log_kwargs)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # parsing data
    test_df = pd.read_json(args.test_file, lines=True)

    print("=====DATA CLEANING=====")
    test_df = test_df.drop_duplicates(subset='title', keep=False).drop_duplicates(subset='maintext', keep=False)

    dropout_symbol = [" ", "\n"]
    for s in dropout_symbol:
        test_df["maintext"] = test_df["maintext"].str.replace(s, "")

    # Adding prefix text to the input, which helps the model to understand the fine-tuning task objective
    test_df["maintext"] = '<idf.lang>' + test_df["maintext"]
    print(f"=====CLEANED VALID DATA===== | Shape:{test_df.shape}")
    print(test_df.head())

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

    model.eval()
    beam_config = GenerationConfig(max_length=args.max_output_length,
                                num_beams=10,
                                length_penalty=1,
                                no_repeat_ngram_size=2,
                                early_stopping=True)
    beam = lambda input_tokens: model.generate(input_tokens, generation_config=beam_config)

    print(f"using generation strategy : beam search")
    total_test_steps = math.ceil(test_df.shape[0])
    progress_bar = tqdm(range(total_test_steps), disable=not accelerator.is_local_main_process)

    pdt_lst, lb_lst = [], []
    for idx, feature in enumerate(test_df.iloc):
        input_tokens = encode_main_text(feature["maintext"], args.max_input_length)
        input_tokens = input_tokens.unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = beam(input_tokens)
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

    result_dict = [{"title": title_, "id": str(feature_["id"])} for feature_, title_ in zip(test_df.iloc, pdt_lst)]

    with open(args.result_file, "w") as f:
        for result_pair in result_dict:
            json.dump(result_pair, f)
            f.write("\n")

if __name__ == "__main__":
    main()