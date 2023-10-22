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
    MT5ForConditionalGeneration
    get_scheduler, # Not sure which to chose
    get_linear_schedule_with_warmup,
)
from tw_rouge import get_rouge

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

logger = get_logger(__name__)

# parsing data