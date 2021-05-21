# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch.nn import CrossEntropyLoss

"""
Loss functions for training different models defined in models.py
"""

ce_loss_sum = CrossEntropyLoss(reduction="sum", ignore_index=0)
ce_loss_mean = CrossEntropyLoss(reduction="mean", ignore_index=0)
ce_loss_no_reduction = CrossEntropyLoss(reduction="none", ignore_index=0)

def base_lstm_train_loss(pred_logits, batch):
    """Calculates CE loss ignoring metadata tokens """

    labels = batch["output"].long()

    ignore_len = 0 if "md" not in batch else 1

    # reshaping to ignore meta data tokens
    pred_logits = pred_logits[:, ignore_len:, :]
    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    mean_loss = ce_loss_mean(pred_logits, labels)
    return mean_loss

def base_lstm_dev_loss(pred_logits, batch):
    """Calculates CE loss ignoring metadata tokens """

    labels = batch["output"].long()

    ignore_len = 0 if "md" not in batch else 1
    n_tokens = torch.sum(labels > 0)

    # reshaping to ignore meta data tokens
    pred_logits = pred_logits[:, ignore_len:, :]
    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    sum_loss = ce_loss_sum(pred_logits, labels)
    return sum_loss, n_tokens

def base_lstm_no_reduction_loss(pred_logits, batch):
    """Calculates CE loss ignoring metadata tokens """

    labels = batch["output"].long()

    ignore_len = 0 if "md" not in batch else 1
    n_tokens = torch.sum(labels > 0, dim=-1)

    # reshaping to ignore meta data tokens
    pred_logits = pred_logits[:, ignore_len:, :]
    initial_pred_shape = pred_logits.shape[:-1]
    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    all_loss = ce_loss_no_reduction(pred_logits, labels)
    all_loss = all_loss.view(initial_pred_shape)

    return all_loss, n_tokens

def advanced_lstm_train_loss(pred_logits, batch):
    """ LSTM loss used by advanced baselines - no need to remove md logits """

    labels = batch["output"].long()
    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    mean_loss = ce_loss_mean(pred_logits, labels)
    return mean_loss

def advanced_lstm_dev_loss(pred_logits, batch):
    """ LSTM loss used by advanced baselines - no need to remove md logits """

    labels = batch["output"].long()
    n_tokens = torch.sum(labels > 0)

    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    sum_loss = ce_loss_sum(pred_logits, labels)
    return sum_loss, n_tokens

def advanced_lstm_no_reduction_loss(pred_logits, batch):
    """ LSTM loss used by advanced baselines - no need to remove md logits """

    labels = batch["output"].long()
    n_tokens = torch.sum(labels > 0, dim=-1)

    initial_pred_shape = pred_logits.shape[:-1]
    pred_logits = torch.reshape(pred_logits, [-1, pred_logits.size(-1)])
    labels = torch.reshape(labels, [-1]) #same as flatten
    all_loss = ce_loss_no_reduction(pred_logits, labels)
    all_loss = all_loss.view(initial_pred_shape)
    return all_loss, n_tokens

##### Utility Wrapper  #####


LOSS_MAP = {"base_lstm": (base_lstm_train_loss, base_lstm_dev_loss),
            "concat_lstm": (advanced_lstm_train_loss, advanced_lstm_dev_loss),}

LOSS_MAP_NO_REDUCTION = {"base_lstm": base_lstm_no_reduction_loss,
                         "concat_lstm": advanced_lstm_no_reduction_loss,}

def get_loss_fn(model_type):
    """Given a model type maps that to the model train and dev loss functions."""
    assert(model_type in LOSS_MAP), f"Invalid model: {model_type}"
    return LOSS_MAP[model_type]

def get_no_reduction_loss_fn(model_type):
    """
    Given a model type maps that to the model loss function with no
    no reduction. Used primarily for per utterance ppl and wer inference.
    """
    assert(model_type in LOSS_MAP_NO_REDUCTION), f"Invalid model: {model_type}"
    return LOSS_MAP_NO_REDUCTION[model_type]
