# This python file is to parse all the arguments needed
# Setting the Hyper-Parameters
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Finetune a mT5 model on Title Generation")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1)
    parser.add_argument("--model_name_or_path", type=str, default="google/mt5-small")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4125252)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
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
        # the pre-trained model should be google/mt5-small
        assert args.model_name_or_path == "google/mt5-small", "the pre-trained model should be google/mt5-small"
    return args
        

