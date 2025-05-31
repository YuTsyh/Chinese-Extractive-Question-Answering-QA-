#!/usr/bin/env python
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0.dev0")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args(cmd_args=None): # <-- Add cmd_args=None here
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args(cmd_args)

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    try:
        get_ipython() # This line won't cause an error in iPython/Jupyter
        print("Running in Colab/Jupyter environment, using predefined arguments.")
        # Define parameters manually in Colab
        colab_args = [
            '--model_name_or_path', 'bert-base-chinese',
            '--train_file', '/mc_train.json',      # Use absolute path
            '--validation_file', '/mc_valid.json', # Use absolute path
            '--output_dir', './paragraph_selector_output_colab',
            '--max_seq_length', '512',
            '--per_device_train_batch_size', '1', # Adjusted for potentially lower memory
            '--per_device_eval_batch_size', '1',  # Adjusted for potentially lower memory
            '--learning_rate', '3e-5',
            '--num_train_epochs', '1',
            '--pad_to_max_length',
        ]
        args = parse_args(colab_args) # args is set here for Colab
    except NameError:
        # Not in an iPython environment, parse arguments from the command line as usual
        print("Running as a standard Python script, parsing command-line arguments.")
        args = parse_args() # args is set here for standard script execution

    # Sending telemetry.
    send_example_telemetry("run_swag_no_trainer", args)

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

    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        print(f"Loading dataset with extension: {extension}, data_files: {data_files}")
        raw_datasets = load_dataset(extension, data_files=data_files)

    print(f"Raw datasets loaded: {raw_datasets}")
    if "train" in raw_datasets:
        print(f"Number of raw train examples: {len(raw_datasets['train'])}")
        if len(raw_datasets['train']) > 0:
            print(f"First raw train example: {raw_datasets['train'][0]}")
    if "validation" in raw_datasets:
        print(f"Number of raw validation examples: {len(raw_datasets['validation'])}")

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    elif raw_datasets["validation"] is not None: # Fallback if only validation is present
        column_names = raw_datasets["validation"].column_names
    else:
        # Fallback if no train or validation data is loaded (e.g. only test data)
        # This might need adjustment if you intend to run predict-only mode
        # For now, assume we need some columns for .map to know what to remove.
        # If mc_*.json files are loaded, these would be the typical columns.
        column_names = ["id", "question_context", "choice_0", "choice_1", "choice_2", "choice_3", "label"]
        logger.warning(f"No train or validation data in raw_datasets to infer column names. Defaulting to: {column_names}")


    # These are the field names in YOUR mc_*.json files
    question_field = "question_context"
    choice_fields = [f"choice_{i}" for i in range(4)]
    label_field = "label"

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code) # Corrected: use args.config_name
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config, trust_remote_code=args.trust_remote_code)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        num_choices = 4
        first_sentences = []
        second_sentences = []
        for i in range(len(examples[question_field])):
            question = examples[question_field][i]
            for j in range(num_choices):
                choice_text = examples[choice_fields[j]][i]
                first_sentences.append(question)
                second_sentences.append(choice_text)
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        tokenized_inputs = {
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized_examples.items()
        }
        tokenized_inputs["labels"] = examples[label_field]
        return tokenized_inputs

    print("Starting dataset mapping (preprocess_function)...")
    with accelerator.main_process_first():
        # Determine columns to remove from the raw_datasets based on its actual content
        # This should be the columns of your mc_*.json files
        actual_columns_in_raw_data = []
        if "train" in raw_datasets and raw_datasets["train"] is not None and len(raw_datasets["train"]) > 0:
             actual_columns_in_raw_data = raw_datasets["train"].column_names
        elif "validation" in raw_datasets and raw_datasets["validation"] is not None and len(raw_datasets["validation"]) > 0:
             actual_columns_in_raw_data = raw_datasets["validation"].column_names
        else:
            logger.warning("Could not reliably determine column names from raw_datasets for removal. Ensure your mc_*.json files are loaded correctly.")
            # Fallback to expected names if necessary, but it's better to infer
            actual_columns_in_raw_data = ["id", "question_context", "choice_0", "choice_1", "choice_2", "choice_3", "label"]


        print(f"Columns to remove during map: {actual_columns_in_raw_data}")
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=actual_columns_in_raw_data
        )

    print(f"Dataset mapping finished. Processed datasets: {processed_datasets}")
    if "train" in processed_datasets:
        print(f"Number of processed train examples: {len(processed_datasets['train'])}")
        if len(processed_datasets['train']) > 0:
            print(f"First processed train example: {processed_datasets['train'][0]}")

    train_dataset = None
    eval_dataset = None

    if args.train_file and "train" in processed_datasets:
        train_dataset = processed_datasets["train"]
        print(f"Train dataset assigned. Length: {len(train_dataset)}")
    else:
        print("Train dataset not assigned (no train_file or 'train' not in processed_datasets).")

    if args.validation_file and "validation" in processed_datasets:
        eval_dataset = processed_datasets["validation"]
        print(f"Eval dataset assigned. Length: {len(eval_dataset)}")
    else:
        print("Eval dataset not assigned (no validation_file or 'validation' not in processed_datasets).")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt"
        )

    train_dataloader = None
    if train_dataset:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        print(f"Train DataLoader created. Number of batches: {len(train_dataloader)}")
    else:
        print("Train DataLoader not created because train_dataset is None.")

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        print(f"Eval DataLoader created. Number of batches: {len(eval_dataloader)}")
    else:
        print("Eval DataLoader not created because eval_dataset is None.")

    if not train_dataloader:
        logger.error("Training dataloader is not available. Cannot proceed with training.")
        return # Exit if no training data

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

    device = accelerator.device
    model.to(device)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes, # Corrected based on original script for multi-GPU
    )

    # Prepare everything with our `accelerator`.
    # Handle eval_dataloader being None
    if eval_dataloader:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else: # If no eval_dataloader, don't pass it to prepare
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print(f"Recalculated max_train_steps: {args.max_train_steps}, num_train_epochs: {args.num_train_epochs}")


    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("swag_no_trainer", experiment_config)

    metric = evaluate.load("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader) # Corrected: use actual train_dataloader length
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader) # Corrected

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir_step = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir_step = os.path.join(args.output_dir, output_dir_step)
                    accelerator.save_state(output_dir_step)

            if completed_steps >= args.max_train_steps:
                break

        # Evaluation
        if eval_dataloader: # Only evaluate if eval_dataloader exists
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            accelerator.print(f"epoch {epoch}: {eval_metric}")

            if args.with_tracking:
                log_metrics = {
                    "accuracy": eval_metric['accuracy'], # Ensure key exists
                    "train_loss": total_loss.item() / len(train_dataloader) if args.with_tracking else None, # Guard against total_loss not defined
                    "epoch": epoch,
                    "step": completed_steps,
                }
                # Filter out None values before logging
                log_metrics = {k: v for k, v in log_metrics.items() if v is not None}
                accelerator.log(log_metrics, step=completed_steps)
        else:
            accelerator.print(f"epoch {epoch}: No evaluation performed as validation data is not available.")


        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir_epoch = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir_epoch = os.path.join(args.output_dir, output_dir_epoch)
            accelerator.save_state(output_dir_epoch)

        if completed_steps >= args.max_train_steps:
            break


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id, # repo_id needs to be defined if push_to_hub is True
                    repo_type="model",
                    token=args.hub_token,
                )
            # Save final eval metric if available
            if 'eval_metric' in locals() and eval_metric: # Check if eval_metric was computed
                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)
            else:
                logger.info("No final evaluation metrics to save as evaluation was not performed or eval_metric is empty.")


    accelerator.wait_for_everyone() # Ensure all processes are done
    if hasattr(accelerator, 'end_training'): # Some versions might not have this
        accelerator.end_training()


if __name__ == "__main__":
    main()