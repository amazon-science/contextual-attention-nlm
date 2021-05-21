# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import torch
from torch.utils.data import IterableDataset
from itertools import cycle
from collections import defaultdict
"""
Custom iterable dataset for streaming in data and data processing utils.
"""

def list_files(directory, ignore_str="json"):
    # ignore_str set to json to skip nbest files
    files = [os.path.join(directory, file) for file in os.listdir(directory) if ignore_str not in file\
                                                                             and os.path.isfile(os.path.join(directory,file))]
    return files

def return_split(file_name):
    split = file_name.split('/')[-1]
    return split

def get_next_utterance(directory, sort_by_function=return_split):
    ''' A generator that yields the next utterance '''
    data_files = list_files(directory)
    data_files.sort(key=sort_by_function, reverse=True)

    for idx, file_path in enumerate(data_files):
        with open(file_path, "r") as transcription_fp:
            for line in transcription_fp:
                yield line

def custom_collate(batch):
    """Collate function to deal with variable length input """
    batch_size = len(batch)
    max_len = max([sample["text_len"] for sample in batch])

    # IMPORTANT: Enforce padding token to be 0
    padded_input = torch.zeros((batch_size, max_len))
    padded_output = torch.zeros((batch_size, max_len))
    text_len = []

    md, md_len = defaultdict(list), defaultdict(list)

    for idx, sample in enumerate(batch):
        curr_len = sample["text_len"]
        text_len.append(curr_len)
        padded_input[idx, :curr_len] = sample["input"]
        padded_output[idx, :curr_len] = sample["output"]

        sample_md = sample["md"]
        sample_md_len = sample["md_len"]
        if sample_md is None:
            md = None
            md_len = None
            continue

        for curr_md_transform, curr_md in sample_md.items():
            md[curr_md_transform].append(curr_md)

        for curr_md_transform, curr_md_len in sample_md_len.items():
            md_len[curr_md_transform].append(curr_md_len)


    text_len = torch.stack(text_len)

    if md:
        for curr_md_transform in md.keys():
            md[curr_md_transform] = torch.stack(md[curr_md_transform])
        for curr_md_transform in md.keys():
            md_len[curr_md_transform] = torch.stack(md_len[curr_md_transform])

    processed_batch = {"input": padded_input,
                       "output": padded_output,
                       "md": md,
                       "text_len": text_len,
                       "md_len": md_len}

    return processed_batch


class MetaDataset(IterableDataset):
    """Dataset that can include meta data information. """

    def __init__(self, data_directory, tokenizer, md_transformer):
        self.data_directory = data_directory
        self.tokenizer = tokenizer
        self.md_transformer = md_transformer
        self.cycle_data = "train" in data_directory

    def generate_processed_stream(self):
        for utterance in get_next_utterance(self.data_directory):
            md_dict, text = self.md_transformer.parse_raw_input(utterance)
            input = torch.tensor(self.tokenizer.encode_text(text, add_sos=True))
            output = torch.tensor(self.tokenizer.encode_text(text, add_eos=True))
            text_len = torch.tensor(len(input))

            if md_dict:
                md = {}
                md_len = {}
                for curr_md_transform, curr_md in md_dict.items():
                    if not isinstance(curr_md, torch.Tensor):
                        curr_md = torch.tensor(self.tokenizer.encode_text(curr_md))
                        curr_md_len = torch.tensor(len(curr_md))
                    else:
                        curr_md_len = torch.tensor(1)

                    md[curr_md_transform] = curr_md
                    md_len[curr_md_transform] = curr_md_len

            else:
                md = None
                md_len = None

            sample = {"input": input,
                      "output": output,
                      "md": md,
                      "text_len": text_len,
                      "md_len": md_len}
            yield sample

    def __iter__(self):
        if self.cycle_data:
            return cycle(self.generate_processed_stream())
        else:
            return self.generate_processed_stream()
