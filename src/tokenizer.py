# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import pickle
import os
import logging
from collections import Counter
from src.data import get_next_utterance, list_files
from abc import ABCMeta, abstractmethod

import sentencepiece as spm

"""
Standard tokenization classes
"""

# NOTE: Keep this as is - padding token id must be 0
BASE_TOKENS = {"</s>":1, "<s>":2, "<unk>": 3, "<pad>": 0}
logger = logging.getLogger(__name__)

class BaseTokenizer(metaclass=ABCMeta):
    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def encode_text(self, input_sentence):
        """Encodes an utterance """
        pass

    @abstractmethod
    def encode(self, input_toks):
        """ Encodes a list of tokens to a list of ids """
        pass

    @abstractmethod
    def decode(self, input_ids):
        """ Decodes a list of ids to a list of tokens """
        pass

    @abstractmethod
    def add_special_tokens(self, special_tokens):
        """Adds a set of special tokens to the base """
        pass

    @classmethod
    @abstractmethod
    def load_tokenizer(cls, tokenizer_path):
        pass

class SPTokenizer(BaseTokenizer):
    def __init__(self, data_path, md_transformer, vocab_limit):
        """ Sentence Piece Tokenizer """
        self.data_path = data_path
        self.md_transformer = md_transformer
        self.vocab_limit = vocab_limit
        self.tokenizer_model = self._generate_model()
        self.special_tokens = BASE_TOKENS

    def _process_data_files(self, dir_path):
        """
        Reads in data in self.data_path and writes out to utterance text to
        files.
        """
        line_count = 0
        file_count = 0

        curr_out_file_name = os.path.join(dir_path, f"processed_{file_count}.txt")
        out_file = open(curr_out_file_name, "w")
        for utterance in get_next_utterance(self.data_path):
            _, text = self.md_transformer.parse_raw_input(utterance)
            line_count += 1

            if (line_count % 20_000 == 0):
                line_count = 0
                file_count += 1
                curr_out_file_name = os.path.join(dir_path,\
                                                  f"processed_{file_count}.txt")
                out_file.close()
                out_file = open(curr_out_file_name, "w")

            out_file.write(text + '\n')

    def _generate_model(self):
        """
        Creates a dataset of processed text files, and trains a sentence piece
        model.
        """
        dir_path = "cache/sp_tokenizer_data"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            self._process_data_files(dir_path)
        data_files = list_files(dir_path)

        model_cache_prefix = f'cache/sp_tokenizer_{self.vocab_limit}'
        spm.SentencePieceTrainer.train(input=data_files,
                                       model_prefix=model_cache_prefix,
                                       vocab_size=self.vocab_limit,
                                       bos_id=BASE_TOKENS["<s>"],
                                       eos_id=BASE_TOKENS["</s>"],
                                       unk_id=BASE_TOKENS["<unk>"],
                                       pad_id=BASE_TOKENS["<pad>"])
        model = spm.SentencePieceProcessor(model_file=f'{model_cache_prefix}.model')
        return model

    def print_special_token_ids(self):
        logger.info("Special Token: Token ID")
        for token, id in self.special_tokens:
            logger.info(f"\t {token}: {id}")

    def add_special_tokens(self, special_tokens):
        logger.info(f"Using special tokens: {special_tokens}")
        curr_vocab_size = self.get_vocab_size()
        for idx, tok in enumerate(set(special_tokens)):
            self.special_tokens[tok] = idx + curr_vocab_size

    def get_vocab_size(self):
        """vocab_limit includes the base_tokens"""
        return self.vocab_limit + len(self.special_tokens) - len(BASE_TOKENS)

    def encode_text(self, input_sentence, add_eos=False, add_sos=False):
        """Encodes an utterance """
        output_ids = []
        input_toks = input_sentence.split()
        if any(input_tok in self.special_tokens for input_tok in input_toks):
            for tok in input_toks:
                id = self.special_tokens[tok]
                output_ids.append(id)
        else:
            output_ids = self.encode(input_sentence, add_eos=add_eos, add_sos=add_sos)

        return output_ids

    def encode(self, input_toks, add_eos=False, add_sos=False):
        """ Encodes a list of tokens to a list of ids """
        return self.tokenizer_model.encode(input_toks,
                                      add_bos=add_sos,
                                      add_eos=add_eos)

    def decode(self, input_ids):
        """ Decodes a list of ids to a list of tokens """
        return self.tokenizer_model.decode(input_ids)

    @classmethod
    def load_tokenizer(cls, tokenizer_path):
        return pickle.load(open(tokenizer_path, "rb"))

class BasicTokenizer(BaseTokenizer):
    def __init__(self, data_path, md_transformer, vocab_limit=None):
        """ Basic Tokenizer."""
        self.data_path = data_path
        self.md_transformer = md_transformer

        self._tok2id = {}
        self._id2tok = {}
        self.vocab = set()
        self.special_tokens = BASE_TOKENS

        self.vocab_limit = vocab_limit
        self.vocab_counter = Counter()

        # Initial construction of class
        self._create_vocab(data_path)
        self._create_token_to_id_map()
        self._create_id_to_token_map()

    def _create_vocab(self, data_path):
        logger.info("Creating vocab for tokenizer")

        for utterance in get_next_utterance(data_path):
            _, text = self.md_transformer.parse_raw_input(utterance)
            tokens = text.split()

            self.vocab_counter.update(tokens)

        if self.vocab_limit:
            word_counts = self.vocab_counter.most_common(self.vocab_limit)
            words, _ = zip(*word_counts)
        else:
            words = self.vocab_counter.elements()
        self.vocab = set(words)

    def _create_token_to_id_map(self):
        for tok, id in self.special_tokens.items():
            self._tok2id[tok] = id
        special_tok_offset = len(self.special_tokens)
        for id, tok in enumerate(self.vocab):
            self._tok2id[tok] = special_tok_offset+id

    def _create_id_to_token_map(self):
        self._id2tok = {val:key for key, val in self._tok2id.items()}

    def get_vocab_size(self):
        """Return vocab + number of special tokens """
        return len(self._tok2id)

    def print_special_token_ids(self):
        logger.info("Special Token: Token ID")
        for token in self.special_tokens:
            id = self._tok2id[token]
            logger.info(f"\t {token}: {id}")

    def add_special_tokens(self, special_tokens):
        # Expanding _tok2id and _id2tok with new special_tokens
        logger.info(f"Using special tokens: {special_tokens}")
        last_id = len(self._tok2id)
        new_tok2id = {}
        for idx, tok in enumerate(set(special_tokens)):
            new_tok2id[tok] = idx+last_id
        new_id2tok = {val:key for key, val in new_tok2id.items()}
        self._tok2id = {**self._tok2id, **new_tok2id}
        self._id2tok = {**self._id2tok, **new_id2tok}

        self.special_tokens =  {**self.special_tokens, **new_tok2id}
        self.print_special_token_ids()

    def encode_text(self, input_sentence, add_eos=False, add_sos=False):
        """Encodes an utterance, and optionally prepends or postpends eos/sos tokens."""
        input_toks = input_sentence.split()
        ids = self.encode(input_toks)
        if add_eos:
            eos_id = self._tok2id["</s>"]
            ids.append(eos_id)
        if add_sos:
            sos_id = self._tok2id["<s>"]
            ids.insert(0, sos_id)
        return ids

    def encode(self, input_toks):
        """ Encodes a list of tokens to a list of ids """
        output_ids = []
        for tok in input_toks:
            if tok in self._tok2id.keys():
                id = self._tok2id[tok]
            else:
                id = self._tok2id["<unk>"]
            output_ids.append(id)
        return output_ids

    def decode(self, input_ids):
        """ Decodes a list of ids to a list of tokens """
        output_toks = []
        for id in input_ids:
            tok = self._id2tok[id]
            output_toks.append(tok)
        return output_toks

    @classmethod
    def load_tokenizer(cls, tokenizer_path):
        return pickle.load(open(tokenizer_path, "rb"))

##### Utility Wrapper  #####

TOKENIZER_MAP = {"basic_tokenizer": BasicTokenizer,
                 "sentence_piece_tokenizer": SPTokenizer}

def get_tokenizer(tokenizer_type, data_path, md_transformer,
                  vocab_limit=0, force_new_creation=False):
    """Either loads in a pretrained tokenizer or creates a new one and saves it"""
    assert(tokenizer_type in TOKENIZER_MAP),\
        f"Invalid tokenizer_type: {tokenizer_type}"
    tokenizer_class = TOKENIZER_MAP[tokenizer_type]
    if vocab_limit:
        saved_tokenizer_path = f"cache/{tokenizer_type}_{vocab_limit}.pkl"
    else:
        saved_tokenizer_path = f"cache/{tokenizer_type}_full.pkl"
    if os.path.exists(saved_tokenizer_path) and not force_new_creation:
        logger.info(f"Loading in saved tokenizer from: {saved_tokenizer_path}")
        tokenizer = tokenizer_class.load_tokenizer(saved_tokenizer_path)
    else:
        if not os.path.exists("cache"):
            os.mkdir("cache")
        logger.info(f"Creating new tokenizer")
        tokenizer = tokenizer_class(data_path, md_transformer, vocab_limit)
        pickle.dump(tokenizer, open(saved_tokenizer_path, "wb"))
        logger.info(f"Saving out tokenizer to: {saved_tokenizer_path}")
    logger.info(f"Size of vocab: {tokenizer.get_vocab_size()}")
    return tokenizer
