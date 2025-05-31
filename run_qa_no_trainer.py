#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

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
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions # Make sure utils_qa.py is in the same directory

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0.dev0") # Ensure your transformers version meets this

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)
    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f: # Added encoding
        json.dump(results, f, indent=4, ensure_ascii=False) # Added ensure_ascii=False


def parse_args(cmd_args_list=None): # Added cmd_args_list for Colab compatibility
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.") # Added for clarity, though script logic might infer
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.") # Added for clarity
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data.")
    parser.add_argument(
        "--max_seq_length", type=int, default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=False
    )
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--doc_stride", type=int, default=128, help="When splitting up a long document into chunks how much stride to take between chunks.")
    parser.add_argument("--n_best_size", type=int, default=20, help="The total number of n-best predictions to generate when looking for an answer.")
    parser.add_argument(
        "--null_score_diff_threshold", type=float, default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument("--version_2_with_negative", action="store_true", help="If true, some of the examples do not have an answer.")
    parser.add_argument("--max_answer_length", type=int, default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_predict_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of prediction examples to this")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES)
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.")
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args(cmd_args_list) # Use cmd_args_list

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.test_file is None:
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        valid_extensions = ["csv", "json"]
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in valid_extensions, f"`train_file` should be a csv or a json file, got {extension}"
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in valid_extensions, f"`validation_file` should be a csv or a json file, got {extension}"
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in valid_extensions, f"`test_file` should be a csv or a json file, got {extension}"

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    # Modifications for Colab/Jupyter
    try:
        get_ipython()
        print("Running in Colab/Jupyter environment, using predefined QA arguments.")
        colab_args = [
            '--model_name_or_path', 'bert-base-chinese',
            '--train_file', '/content/qa_train.json', 
            '--validation_file', '/content/qa_valid.json',
            '--output_dir', './qa_model_output_colab',
            '--max_seq_length', '384',
            '--doc_stride', '128',
            '--per_device_train_batch_size', '2', # Adjusted for typical QA memory usage
            '--per_device_eval_batch_size', '4',
            '--learning_rate', '3e-5',
            '--num_train_epochs', '1', # Start with 1 for testing
            '--pad_to_max_length',
            '--preprocessing_num_workers', '2',
            '--do_train', # Explicitly enable training
            '--do_eval',  # Explicitly enable evaluation
        ]
        args = parse_args(colab_args)
    except NameError:
        print("Running as a standard Python script, parsing command-line arguments.")
        args = parse_args() # This will use sys.argv
    
    send_example_telemetry("run_qa_no_trainer", args)

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

    if args.seed is not None:
        set_seed(args.seed)

    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                if args.output_dir is None:
                     raise ValueError("Need either a `hub_model_id` or `output_dir` for pushing to hub.")
                repo_name = Path(args.output_dir).name
            else:
                repo_name = args.hub_model_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            with open(os.path.join(args.output_dir, ".gitignore"), "w+", encoding="utf-8") as gitignore:
                if "step_*" not in gitignore: gitignore.write("step_*\n")
                if "epoch_*" not in gitignore: gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code)
    else:
        data_files = {}
        extension = None # Initialize extension
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            if extension is None: extension = args.validation_file.split(".")[-1] # Set if not already
        if args.test_file is not None:
            data_files["test"] = args.test_file
            if extension is None: extension = args.test_file.split(".")[-1] # Set if not already
        
        if not extension: # If no files provided, extension remains None
            raise ValueError("No data files provided, cannot determine dataset extension.")

        logger.info(f"Loading dataset with extension: {extension}, data_files: {data_files}")
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    
    logger.info(f"Raw datasets features: {raw_datasets['train'].features if 'train' in raw_datasets else 'No train set'}")


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script.")

    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, trust_remote_code=args.trust_remote_code
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config, trust_remote_code=args.trust_remote_code)

    # Standard SQuAD column names for features that prepare_train/validation_features will use
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers" 

    # Column names from the SQuAD-flattened dataset that we want to remove after tokenization by .map()
    # The `datasets` SQuAD loader flattens the JSON into these columns.
    column_names_to_remove_in_map = ['id', 'title', 'context', 'question', 'answers']


    pad_on_right = tokenizer.padding_side == "right"
    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else (input_ids.index(tokenizer.bos_token_id) if tokenizer.bos_token_id in input_ids else 0)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("Training requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names_to_remove_for_map, # Use the defined list
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if args.max_train_samples is not None: # Re-apply select if map changed number of samples (due to overflow)
            if len(train_dataset) > args.max_train_samples:
                 train_dataset = train_dataset.select(range(args.max_train_samples))


    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("Evaluation requires a validation dataset.")
        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        with accelerator.main_process_first():
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names_to_remove_for_map, # Use the defined list
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if args.max_eval_samples is not None: # Re-apply select
            if len(eval_dataset) > args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))


    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("Prediction requires a test dataset.")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                prepare_validation_features, # Uses the same feature prep as validation
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names_to_remove_for_map, # Use the defined list
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if args.max_predict_samples is not None: # Re-apply select
            if len(predict_dataset) > args.max_predict_samples:
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))


    if args.do_train:
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))): # Handle small datasets
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        pad_to_multiple_of = 8 if accelerator.mixed_precision != "no" else None
        if accelerator.mixed_precision == "fp8": pad_to_multiple_of = 16
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    train_dataloader = None
    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    
    eval_dataloader = None
    if args.do_eval:
        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

    predict_dataloader = None
    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
    
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        for i, output_logit in enumerate(start_or_end_logits):
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size <= len(dataset): # Corrected boundary condition
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
            step += batch_size
        return logits_concat

    if args.do_train:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        else: # If max_train_steps is set, override num_epochs
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes, # Adjusted for multi-GPU
        )

        # Prepare for training
        if eval_dataloader:
             model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
            )
        else:
             model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler
            )


        # Recalculate after prepare
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_update_steps_per_epoch > 0: # Avoid division by zero
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        else: # No steps if dataloader is empty
            args.num_train_epochs = 0
            args.max_train_steps = 0


        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        if args.with_tracking:
            experiment_config = vars(args)
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("qa_no_trainer", experiment_config)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process or args.max_train_steps == 0)
        completed_steps = 0
        starting_epoch = 0
        total_loss_for_epoch = 0.0 # For tracking average loss per epoch

        if args.resume_from_checkpoint:
            if os.path.isdir(args.resume_from_checkpoint): # Check if it's a directory
                checkpoint_path = args.resume_from_checkpoint
                path_basename = os.path.basename(args.resume_from_checkpoint)
                accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
                accelerator.load_state(checkpoint_path)
                training_difference = os.path.splitext(path_basename)[0]
                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                    if num_update_steps_per_epoch > 0: completed_steps = starting_epoch * num_update_steps_per_epoch
                else:
                    resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                    if len(train_dataloader) > 0:
                        starting_epoch = resume_step // len(train_dataloader)
                        resume_step -= starting_epoch * len(train_dataloader)
                    else: starting_epoch = 0; resume_step = 0
                    if args.gradient_accumulation_steps > 0 : completed_steps = resume_step // args.gradient_accumulation_steps
                progress_bar.update(completed_steps)
            else:
                logger.warning(f"Resume from checkpoint path {args.resume_from_checkpoint} not found or not a directory. Starting from scratch.")


        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss_for_epoch = 0.0
            active_dataloader = train_dataloader
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
                resume_step = None # Consume resume_step

            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss_for_epoch += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                        output_dir_step = f"step_{completed_steps}"
                        if args.output_dir is not None: output_dir_step = os.path.join(args.output_dir, output_dir_step)
                        accelerator.save_state(output_dir_step)
                
                if completed_steps >= args.max_train_steps: break
            
            avg_epoch_loss = total_loss_for_epoch / len(active_dataloader) if len(active_dataloader) > 0 else 0.0
            logger.info(f"Epoch {epoch} finished. Average Training Loss: {avg_epoch_loss}")


            if args.checkpointing_steps == "epoch":
                output_dir_epoch = f"epoch_{epoch}"
                if args.output_dir is not None: output_dir_epoch = os.path.join(args.output_dir, output_dir_epoch)
                accelerator.save_state(output_dir_epoch)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                if accelerator.is_main_process and repo_id:
                    tokenizer.save_pretrained(args.output_dir)
                    api.upload_folder(commit_message=f"Training in progress epoch {epoch}", folder_path=args.output_dir, repo_id=repo_id, repo_type="model", token=args.hub_token)
            
            if completed_steps >= args.max_train_steps: break
    
    # Evaluation
    eval_metric = {} # Initialize
    if args.do_eval:
        if not eval_dataloader:
            logger.warning("Evaluation requested but no eval_dataloader available.")
        else:
            logger.info("***** Running Evaluation *****")
            logger.info(f"  Num examples = {len(eval_examples)}") # Use eval_examples for original count
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
            all_start_logits, all_end_logits = [], []
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad(): outputs = model(**batch)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                if not args.pad_to_max_length:
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
            
            if not all_start_logits: # Check if list is empty
                 logger.warning("No logits collected for evaluation. Skipping metric computation.")
            else:
                max_len = max(x.shape[1] for x in all_start_logits)
                start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len) # eval_dataset here is processed
                end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
                del all_start_logits, all_end_logits
                outputs_numpy = (start_logits_concat, end_logits_concat)
                # post_processing_function expects eval_examples (original) and eval_dataset (processed features)
                prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy, stage="eval")
                eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
                logger.info(f"Evaluation metrics: {eval_metric}")

    # Prediction
    if args.do_predict:
        if not predict_dataloader:
            logger.warning("Prediction requested but no predict_dataloader available.")
        else:
            logger.info("***** Running Prediction *****")
            logger.info(f"  Num examples = {len(predict_examples)}") # Use predict_examples for original count
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
            all_start_logits, all_end_logits = [], []
            model.eval()
            for step, batch in enumerate(predict_dataloader):
                with torch.no_grad(): outputs = model(**batch)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                if not args.pad_to_max_length:
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

            if not all_start_logits:
                logger.warning("No logits collected for prediction. Skipping metric computation.")
            else:
                max_len = max(x.shape[1] for x in all_start_logits)
                start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len) # predict_dataset here is processed
                end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
                del all_start_logits, all_end_logits
                outputs_numpy = (start_logits_concat, end_logits_concat)
                # post_processing_function expects predict_examples (original) and predict_dataset (processed features)
                prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy, stage="predict")
                # Note: metric.compute for predict usually needs references if you have them, otherwise it's just for formatting
                # If your test_file for prediction doesn't have answers, you can't compute SQuAD metrics directly.
                # You'd typically just save the `prediction.predictions`
                # For now, let's assume you might have references for a "test" set similar to validation.
                try:
                    predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
                    logger.info(f"Predict metrics: {predict_metric}")
                except Exception as e:
                    logger.warning(f"Could not compute metrics for predictions (possibly no references): {e}")
                    logger.info(f"Predictions: {prediction.predictions[:5]}") # Log a few predictions


    if args.with_tracking:
        log_data = {"epoch": epoch if 'epoch' in locals() else args.num_train_epochs -1, "step": completed_steps}
        if eval_metric: log_data["squad_v2" if args.version_2_with_negative else "squad"] = eval_metric
        if args.do_train and 'total_loss_for_epoch' in locals() and len(train_dataloader) > 0: # Check if training was done
            log_data["train_loss"] = total_loss_for_epoch / len(train_dataloader) # Use final epoch loss
        if args.do_predict and 'predict_metric' in locals() and predict_metric:
            log_data["squad_v2_predict" if args.version_2_with_negative else "squad_predict"] = predict_metric
        accelerator.log(log_data, step=completed_steps)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub and repo_id:
                api.upload_folder(commit_message="End of training", folder_path=args.output_dir, repo_id=repo_id, repo_type="model", token=args.hub_token)
            if eval_metric:
                logger.info(json.dumps(eval_metric, indent=4))
                save_prefixed_metrics(eval_metric, args.output_dir)
            elif args.do_eval: # If eval was run but eval_metric is empty (e.g. no logits)
                logger.warning("Evaluation was run, but no metrics were computed or eval_metric is empty.")


if __name__ == "__main__":
    main()