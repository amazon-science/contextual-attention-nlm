# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import logging
import click
import os
import math

from src.tokenizer import get_tokenizer
from src.metadata import MetaDataTransformer
from src.model import get_model
from src.util import move_to_device, device, save_model_checkpoint, eval_model,\
                     get_dataloader, get_lr_scheduler
from src.loss import get_loss_fn, get_no_reduction_loss_fn

from configparser import ConfigParser
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

"""
Basic utils for setting up main training and evaluation loops.
"""

def setup_config(config_file_path):
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(config):
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = os.path.join(config.get("EXPERIMENT", "experiment_directory"), "experiment.log")
    logging.basicConfig(
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(log_file_name),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"Initializing experiment: {config.get('EXPERIMENT', 'experiment_directory')}")
    logging.info(f"Running model on device: {device}")

def setup(config_file_path):
    config = setup_config(config_file_path)
    setup_logger(config)
    return config

"""
Main script for training a language model with meta data input.
"""

@click.command()
@click.argument('config_file_path')
def main(config_file_path):

    ####### Initial Model Setup #######
    config = setup(config_file_path)
    writer = SummaryWriter(os.path.join(config.get("EXPERIMENT", "experiment_directory"), "tb_log"))

    md_transformer = MetaDataTransformer(text_index=config.get("METADATA", "text_index"),
                                         md_indices=config.get("METADATA", "md_indices",
                                                                           fallback=""),
                                         md_transformations=config.get("METADATA",
                                                                       "md_transformations",
                                                                       fallback=""))

    tokenizer = get_tokenizer(tokenizer_type=config.get("TOKENIZER", "tokenizer_type"),
                              data_path=config.get("DATA", "train_data_directory_full"),
                              md_transformer=md_transformer,
                              vocab_limit=int(config.get("TOKENIZER", "vocab_limit")),
                              force_new_creation=False)
    tokenizer.add_special_tokens(md_transformer.get_md_tokens())

    # Constructing datasets
    train_dataloader = get_dataloader(config, tokenizer, md_transformer, "train", "full")
    dev_dataloader_full = get_dataloader(config, tokenizer, md_transformer, "dev", "full")
    dev_dataloader_head = get_dataloader(config, tokenizer, md_transformer, "dev", "head")
    dev_dataloader_tail = get_dataloader(config, tokenizer, md_transformer, "dev", "tail")

    # Loading model
    model = get_model(config, tokenizer.get_vocab_size())
    train_loss_fn, dev_loss_fn = get_loss_fn(config.get("MODEL", "model_type"))

    ####### Training Configurations #######
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    lr = float(config.get("TRAIN", "learning_rate"))
    optimizer = Adam(model_params, lr=lr)
    # Scheduler is either 1) lr_scheduler.ReduceLROnPlateau or 2) lr_scheduler.OneCycleLR
    update_lr_per_step, scheduler = get_lr_scheduler(config, optimizer)

    print_every = int(config.get("TRAIN", "print_every"))
    eval_every = int(config.get("TRAIN", "eval_every"))

    max_train_steps = int(config.get("TRAIN", "max_train_steps"))

    logging.info(f"Training Configuration:")
    logging.info(f"\t Evaluating model every: {eval_every} steps")
    logging.info(f"\t Training with Adam using lr: {lr}")

    ####### Training Loop #######
    model.to(device)
    model.train()
    logging.info("Beginning training loop.")
    best_dev_loss_full = 1e3
    best_dev_loss_head = 1e3
    best_dev_loss_tail = 1e3
    best_dev_loss_full_step = -1
    best_dev_loss_head_step = -1
    best_dev_loss_tail_step = -1

    for train_step, train_batch in enumerate(train_dataloader):
        if train_step > max_train_steps:
            break

        # MODEL EVALUATION
        if (train_step and train_step % eval_every == 0):
            model.eval()

            dev_loss_full, dev_ppl_full = eval_model(model, dev_dataloader_full,
                                                     dev_loss_fn)
            dev_loss_head, dev_ppl_head = eval_model(model, dev_dataloader_head,
                                                     dev_loss_fn)
            dev_loss_tail, dev_ppl_tail = eval_model(model, dev_dataloader_tail,
                                                     dev_loss_fn)

            logging.info(f"\t Finished Evaluation Model.")
            logging.info(f"\t \t Full Dev Data -- Loss: {dev_loss_full}, PPL: {dev_ppl_full}")
            logging.info(f"\t \t Head Dev Data -- Loss: {dev_loss_head}, PPL: {dev_ppl_head}")
            logging.info(f"\t \t Tail Dev Data -- Loss: {dev_loss_tail}, PPL: {dev_ppl_tail}")
            writer.add_scalar('Loss/dev_full', dev_loss_full, train_step)
            writer.add_scalar('PPL/dev_full', dev_ppl_full, train_step)
            writer.add_scalar('Loss/dev_head', dev_loss_head, train_step)
            writer.add_scalar('PPL/dev_head', dev_ppl_head, train_step)
            writer.add_scalar('Loss/dev_tail', dev_loss_tail, train_step)
            writer.add_scalar('PPL/dev_tail', dev_ppl_tail, train_step)

            if dev_loss_full < best_dev_loss_full:
                best_dev_loss_full = dev_loss_full
                best_dev_loss_full_step = train_step
            if dev_loss_head < best_dev_loss_head:
                best_dev_loss_head = dev_loss_head
                best_dev_loss_head_step = train_step
            if dev_loss_tail < best_dev_loss_tail:
                best_dev_loss_tail = dev_loss_tail
                best_dev_loss_tail_step = train_step

            if not update_lr_per_step:
                scheduler.step(dev_loss_full)

            save_model_checkpoint(model, train_step, config,
                                         dev_loss_full, dev_ppl_full,
                                         dev_loss_head, dev_ppl_head,
                                         dev_loss_tail, dev_ppl_tail)


            model.train()

        model.zero_grad()
        train_batch = move_to_device(train_batch)
        pred_logits = model(train_batch)
        train_loss = train_loss_fn(pred_logits, train_batch)

        if (train_step and train_step % print_every == 0):
            logging.info(f"\t Training Step {train_step} Loss: {train_loss.item()}")
            writer.add_scalar('Loss/train', train_loss.item(), train_step)

        train_loss.backward()
        optimizer.step()
        if update_lr_per_step and train_step < max_train_steps:
            # don't step scheduler at last update step
            scheduler.step()

    logging.info(f"Completed {max_train_steps} training steps.")
    logging.info(f"Evaluation Overview: ")
    logging.info(f"Best full dev loss/PPL: {best_dev_loss_full}/{math.exp(best_dev_loss_full)} \t step: {best_dev_loss_full_step}")
    logging.info(f"Best head dev loss/PPL: {best_dev_loss_head}/{math.exp(best_dev_loss_head)} \t step: {best_dev_loss_head_step}")
    logging.info(f"Best tail dev loss/PPL: {best_dev_loss_tail}/{math.exp(best_dev_loss_tail)} \t step: {best_dev_loss_tail_step}")

if __name__ == "__main__":
    main()
