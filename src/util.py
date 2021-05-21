# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import os
import shutil
import json
import subprocess
from src.data import custom_collate, MetaDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

"""Utils for model training and evaluation"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_to_device(batch):
    updated_batch = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            if key not in updated_batch:
                updated_batch[key] = {}
            for sub_key, sub_val in val.items():
                if sub_val is not None:
                    updated_batch[key][sub_key] = sub_val.to(device)
        else:
            if val is not None:
                updated_batch[key] = val.to(device)
    return updated_batch

def get_dataloader(config, tokenizer, md_transformer, partition, split, config_section="DATA"):
    dataset = MetaDataset(config.get(config_section, f"{partition}_data_directory_{split}"),
                          tokenizer,
                          md_transformer)
    dataloader = DataLoader(dataset,
                            batch_size=int(config.get("MODEL", "batch_size")),
                            collate_fn=custom_collate)
    return dataloader

def save_model_checkpoint(model, step, config,
                          dev_loss_full, dev_ppl_full,
                          dev_loss_head, dev_ppl_head,
                          dev_loss_tail, dev_ppl_tail):
    """Saves out model artifact along with basic statistics about checkpoint"""
    experiment_directory = config.get("EXPERIMENT", "experiment_directory")
    checkpoints_dir = os.path.join(experiment_directory, "checkpoints")

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    curr_checkpoint = os.path.join(checkpoints_dir, f"checkpoint_step_{step}")
    os.mkdir(curr_checkpoint)
    with open(os.path.join(curr_checkpoint, "info.txt"), "w") as f_out :
        f_out.write(f"Step {step}\n")
        f_out.write(f"Full Dev Loss: {dev_loss_full} - Dev PPL {dev_ppl_full}\n")
        f_out.write(f"Head Dev Loss: {dev_loss_head} - Dev PPL {dev_ppl_head}\n")
        f_out.write(f"Tail Dev Loss: {dev_loss_tail} - Dev PPL {dev_ppl_tail}")

    torch.save(model, os.path.join(curr_checkpoint, "model.pt"))

def eval_model(model, dataloader, loss_fn):
    """
    Evaluates model on a given data loader. If per_utterance_ppl is set to
    True function returns the ppl of each utterance in the dataloader.
    """
    cumulative_loss = 0.0
    cumulative_tokens = 0
    for step, batch in enumerate(dataloader):

        batch = move_to_device(batch)
        pred_logits = model(batch)
        loss, tokens = loss_fn(pred_logits, batch)

        cumulative_loss += loss.item()
        cumulative_tokens += tokens.item()
    loss = cumulative_loss/cumulative_tokens
    ppl = torch.exp(torch.tensor(loss))
    return (loss, ppl)

def get_lr_scheduler(config, optimizer):
    """
    Returns a bool of (update_lr_per_step, lr_scheduler)

    """
    lr = float(config.get("TRAIN", "learning_rate"))
    scheduler_type = config.get("TRAIN", "scheduler", fallback="plateau")

    if scheduler_type == "plateau":
        eps_tolerance = float(config.get("TRAIN", "eps_tolerance", fallback='0'))
        patience = int(config.get("TRAIN", "patience", fallback='1'))
        decay_factor = float(config.get("TRAIN", "decay_factor", fallback='0.5'))
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=decay_factor,
                                      patience=patience,
                                      eps=eps_tolerance)
        update_lr_per_step = False
    elif scheduler_type == "one_cycle":
        max_train_steps = int(config.get("TRAIN", "max_train_steps"))
        anneal_strategy = config.get("TRAIN", "anneal_strategy", fallback="cos")
        pct_start = float(config.get("TRAIN", "pct_start", fallback=0.3))
        scheduler = OneCycleLR(optimizer,
                               max_lr=lr,
                               total_steps=max_train_steps,
                               anneal_strategy=anneal_strategy,
                               pct_start=pct_start)
        update_lr_per_step = True
    else:
        raise Exception(f"Invalid scheduler type: {scheduler_type}")

    return (update_lr_per_step, scheduler)
