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
from src.util import move_to_device, device, eval_model, get_dataloader
from src.loss import get_loss_fn, get_no_reduction_loss_fn

from configparser import ConfigParser

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
    log_file_name = os.path.join(config.get("EXPERIMENT", "experiment_directory"), "inference_result.log")
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
Main script for running inference on a trained model.
"""

@click.command()
@click.argument('config_file_path')
def main(config_file_path):
    config = setup(config_file_path)
    md_transformer = MetaDataTransformer(text_index=config.get("TEST", "text_index"),
                                         md_indices=config.get("TEST", "md_indices",
                                                                           fallback=""),
                                         md_transformations=config.get("TEST",
                                                                       "md_transformations",
                                                                       fallback=""))

    tokenizer = get_tokenizer(tokenizer_type=config.get("TOKENIZER", "tokenizer_type"),
                              data_path=config.get("DATA", "train_data_directory_full"),
                              md_transformer=md_transformer,
                              vocab_limit=int(config.get("TOKENIZER", "vocab_limit")),
                              force_new_creation=False)
    tokenizer.add_special_tokens(md_transformer.get_md_tokens())

    # Getting dataloaders
    ppl_dataloader_full = get_dataloader(config, tokenizer, md_transformer, "ppl", "full", config_section="TEST")
    ppl_dataloader_head = get_dataloader(config, tokenizer, md_transformer, "ppl", "head", config_section="TEST")
    ppl_dataloader_tail = get_dataloader(config, tokenizer, md_transformer, "ppl", "tail", config_section="TEST")

    # Setting up model
    model = torch.load(config.get("TEST", "model_path"), map_location=device)
    _, dev_loss_fn = get_loss_fn(config.get("MODEL", "model_type"))
    no_reduction_loss_fn = get_no_reduction_loss_fn(config.get("MODEL", "model_type"))

    #### Evaluation Cycle ###
    model.eval()
    model.to(device)

    loss_full, ppl_full = eval_model(model, ppl_dataloader_full,
                                            dev_loss_fn)
    loss_head, ppl_head = eval_model(model, ppl_dataloader_head,
                                            dev_loss_fn)
    loss_tail, ppl_tail = eval_model(model, ppl_dataloader_tail,
                                            dev_loss_fn)

    logging.info("Full evaluation: ")
    logging.info(f"\t loss: {loss_full} ppl: {ppl_full}")
    logging.info("Head evaluation: ")
    logging.info(f"\t loss: {loss_head} ppl: {ppl_head}")
    logging.info("Tail evaluation: ")
    logging.info(f"\t loss: {loss_tail} ppl: {ppl_tail}")

if __name__ == "__main__":
    main()
