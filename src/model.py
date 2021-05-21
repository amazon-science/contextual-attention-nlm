# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.util import device

"""
Attention Model Classes. Note that all classes share the same set of arguments.
"""

class BahdanauAttention(nn.Module):
    '''
    Standard bahdanau attention mechanism: https://arxiv.org/abs/1409.0473
    '''
    def __init__(self, md_dim, query_dim, md_group_size, use_null_token):
        super().__init__()

        hidden_dim = md_dim # TODO (low priority): allow user to customize this
        self.md_dim = md_dim #size of keys
        self.query_dim = query_dim #size of query vector
        self.md_group_size = md_group_size

        self.use_null_token = use_null_token
        if self.use_null_token:
            self.zeros = torch.zeros([1,1,self.md_dim], requires_grad=False)

        self.key_projection = nn.Linear(md_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(query_dim, hidden_dim, bias=False)
        self.energy_projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, input, query):
        if query.dim() == 3:
            # 3-dimensional query means we are precomputing attention
            seq_len = query.size(0)
            query = query.view(seq_len, -1, 1, self.query_dim)
        else:
            query = query.view(-1, 1, self.query_dim)

        input_dim = input.dim()
        if input_dim == 3:
            input = input.view(-1, self.md_group_size, self.md_dim)
        elif input_dim == 4:
            input = input.view(seq_len, -1, self.md_group_size, self.md_dim)
        else:
            raise Exception(f"Invalid number of input dimension: {input_dim}")

        if self.use_null_token:
            if input_dim == 3:
                zeros = self.zeros.repeat(input.size(0), 1, 1).to(device)
                input = torch.cat((input, zeros), dim=1)
            else:
                zeros = self.zeros.repeat(input.size(0), input.size(1), 1, 1).to(device)
                test = self.zeros.repeat(input.size(0), 1, 1).to(device)
                input = torch.cat((input, zeros), dim=2)

        hidden_keys = self.key_projection(input)
        hidden_query = self.query_projection(query)

        scores = self.energy_projection(torch.tanh(hidden_query + hidden_keys))
        alphas = nn.Softmax(dim=-1)(scores).transpose(-1,-2)
        context = torch.matmul(alphas, input).squeeze()
        return context

class GeneralAttention(nn.Module):
    '''
    Implements a general purpose general attention mechanism where key and query
    vectors are multiplied together by a learned weight matrix W.
    '''
    def __init__(self, md_dim, query_dim, md_group_size, use_null_token):

        super().__init__()

        self.md_dim = md_dim
        self.query_dim = query_dim
        self.md_group_size = md_group_size
        self.use_null_token = use_null_token

        if self.use_null_token:
            self.zeros = torch.zeros([1,1,self.md_dim], requires_grad=False)

        self.W = nn.Parameter(torch.Tensor(self.query_dim, self.md_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, input, query):
        if query.dim() == 3:
            # 3-dimensional query means we are precomputing attention
            seq_len = query.size(0)
            query = query.view(seq_len, -1, 1, self.query_dim)
        else:
            query = query.view(-1, 1, self.query_dim)

        input_dim = input.dim()
        if input_dim == 3:
            input = input.view(-1, self.md_group_size, self.md_dim)
        elif input_dim == 4:
            input = input.view(seq_len, -1, self.md_group_size, self.md_dim)
        else:
            raise Exception(f"Invalid number of input dimension: {input_dim}")

        if self.use_null_token:
            if input_dim == 3:
                zeros = self.zeros.repeat(input.size(0), 1, 1).to(device)
                input = torch.cat((input, zeros), dim=1)
            else:
                zeros = self.zeros.repeat(input.size(0), input.size(1), 1, 1).to(device)
                test = self.zeros.repeat(input.size(0), 1, 1).to(device)
                input = torch.cat((input, zeros), dim=2)

        scores = torch.matmul(query, self.W)
        scores = torch.matmul(scores, input.transpose(-1,-2))

        alphas = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(alphas, input).squeeze()
        return context

ATTENTION_MAP = {"general": GeneralAttention, "bahdanau": BahdanauAttention}


class MetadataConstructor(nn.Module):
    """
    General module that processes different metadata and construct a metadata
    representation by using an attention based approach, or more simply concatenating
    different metadata embeddings.
    """

    def __init__(self, metadata_constructor_params, dimension_params):

        super().__init__()

        self.md_projection_dim = metadata_constructor_params["md_projection_dim"]
        self.md_dims = metadata_constructor_params["md_dims"]
        self.md_group_sizes = metadata_constructor_params["md_group_sizes"]
        self.context_dim = dimension_params["context_dim"]

        # when using radians
        self.attention_mechanism = metadata_constructor_params["attention_mechanism"]

        if self.attention_mechanism:
            self.query_type = metadata_constructor_params["query_type"]

            assert(self.attention_mechanism in ATTENTION_MAP),\
                f"Invalid attention type: {self.attention_mechanism}"
            assert(self.query_type in ("word", "hidden")),\
                f"Invalid query type: {self.query_type}"

            query_dim = self.get_query_dim(dimension_params)

            self.use_null_token = metadata_constructor_params["use_null_token"]

            self.attention_modules = []
            for md_dim, md_group_size in zip(self.md_dims, self.md_group_sizes):
                attention_module = ATTENTION_MAP[self.attention_mechanism](md_dim,
                                                                           query_dim,
                                                                           md_group_size,
                                                                           self.use_null_token).to(device)
                self.attention_modules.append(attention_module)

        # After attention module, the resulting metadata embeddings are projected
        # to size md_projection_dim
        self.projection_layers = []
        for md_dim in self.md_dims:
            projection = nn.Linear(md_dim, self.md_projection_dim).to(device)
            self.projection_layers.append(projection)

        # The resulting metadata embeddings can now be combined via another
        # attention mechanism (specified by "hierarchical_attention" bool parameter),
        # or via a simpler concatenation of the metadata together
        self.use_hierarchical_attention = metadata_constructor_params["hierarchical_attention"]
        if self.use_hierarchical_attention:
            num_attention_groups = len(self.md_dims)
            query_dim = self.get_query_dim(dimension_params)
            # NOTE: we use the same query embedding in the attention module
            # as in the previous attention modules
            self.hierarchical_attention_module =  ATTENTION_MAP[self.attention_mechanism](self.md_projection_dim,
                                                                                          query_dim,
                                                                                          num_attention_groups,
                                                                                          self.use_null_token).to(device)
            context_projection_input_dim = self.md_projection_dim
        else:
            # If metadata is not combined hierarchically, all the metadata is
            # instead concated together. Computing the resulting size of the
            # concatenated embedding
            context_projection_input_dim = 0
            for md_group_size in self.md_group_sizes:
                if self.attention_mechanism:
                    context_projection_input_dim += self.md_projection_dim
                else:
                    context_projection_input_dim += self.md_projection_dim * md_group_size

        # Finally the metadata embedding (either concatenated or combined via attetntion)
        #  are projected to size of context_dim
        self.context_projection = nn.Linear(context_projection_input_dim,
                                            self.context_dim).to(device)
        self.context_normalization = nn.LayerNorm(self.context_dim)

    def get_query_dim(self, dimension_params):
        ''' Simple helper function to return size of the query '''
        if self.query_type == "word":
            query_dim = dimension_params["emb_dim"]
        elif self.query_type == "hidden":
            query_dim = dimension_params["hidden_dim"]
        return query_dim

    def is_precomputable(self):
        ''' Logic for determining is attention can be precomputed'''
        if not self.attention_mechanism or self.query_type == "word":
            return True
        else:
            return False

    @staticmethod
    def preprocess_md_util(md_embs, embedding_projection, md_dims, md_group_sizes):
        ''' Static helper to be used by base lstm model '''
        processed_md = {}
        for idx, (md_transform, md) in enumerate(md_embs.items()):
            md_dim = md_dims[idx]
            md_group_size = md_group_sizes[idx]
            # Notice this breaks if md_dim == md_group_size
            if md.shape[-1] != md_dim*md_group_size:
                md = embedding_projection(md)

            processed_md[md_transform] = md
        return processed_md

    def preprocess_md(self, md_embs, embedding_projection):
        ''' Preprocess metadata input to have correct dimensionality '''
        return MetadataConstructor.preprocess_md_util(md_embs, embedding_projection,
                                                      self.md_dims, self.md_group_sizes)

    def concat_md(self, input_md):
        """ Flattens and concatenates input embeddings"""
        mds = []
        for md in input_md:
            if not self.attention_mechanism:
                md = md.flatten(start_dim=1)
            mds.append(md)
        concat_md = torch.cat(mds, -1)
        return concat_md

    def forward(self, md_embs, query=None):
        '''
        Query can be None if attention mechanism is not used
        '''
        processed_mds = []
        for idx, md in enumerate(md_embs.values()):
            if self.attention_mechanism:
                attention_module = self.attention_modules[idx]
                md = attention_module(md, query)

            # Only need to project data if more than one metadata group used
            projection_layer = self.projection_layers[idx]
            processed_md = projection_layer(md)
            processed_mds.append(processed_md)

        if self.use_hierarchical_attention:
            combined_md = self.hierarchical_attention_module(torch.stack(processed_mds), query)
        else:
            combined_md = self.concat_md(processed_mds)

        context_emb = self.context_projection(combined_md)
        context_emb = self.context_normalization(context_emb)
        context_emb = torch.tanh(context_emb)
        return context_emb


'''
Base LSTM models that we are conditioned with non-linguistic context. In the paper
we provide two methods for adding the context to the LSTM model, a concatenation-based
method and a factor-based method. This open-source codebase provides the concatenation-method
since it generally yielded the best results.
'''

class ConcatLSTM(nn.Module):
    """
    Implements the hidden state ConcatCell Approach proposed by Mikolov et al. 2012
    Paper Details: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rnn_ctxt.pdf
    """

    def __init__(self, dimension_params, metadata_constructor_params, layer_params):

        super().__init__()
        self.emb_dim = dimension_params["emb_dim"]
        self.context_dim = dimension_params["context_dim"]
        self.hidden_dim = dimension_params["hidden_dim"]
        self.vocab_size = dimension_params["vocab_size"]

        self.n_layers = layer_params["n_layers"]
        self.use_softmax_adaptation = layer_params["use_softmax_adaptation"]
        self.use_layernorm = layer_params["use_layernorm"]
        self.use_weight_tying = layer_params["use_weight_tying"]

        self.metadata_constructor = MetadataConstructor(metadata_constructor_params,
                                                        dimension_params)

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)

        self.gate_size = 4 * self.hidden_dim # input, forget, gate, output
        self._all_weights = nn.ParameterList()
        self._params_per_layer = 5

        for layer in range(self.n_layers):
            self.layer_input_size = self.emb_dim if layer == 0 else self.hidden_dim

            # weight matrix for meta data
            w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.layer_input_size))
            w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_dim))

            w_mh = nn.Parameter(torch.Tensor(self.gate_size, self.context_dim))

            b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            b_hh = nn.Parameter(torch.Tensor(self.gate_size))

            for param in (w_mh, w_ih, w_hh, b_ih, b_hh):
                self._all_weights.append(param)

        if self.use_softmax_adaptation:
            self.md_vocab_projection = nn.Linear(self.context_dim, self.vocab_size)

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_dim)

        if self.use_weight_tying:
            self.vocab_projection = nn.Linear(self.emb_dim, self.vocab_size)
            self.embedding_projection = nn.Linear(self.hidden_dim, self.emb_dim)
            self.vocab_projection.weight = self.embeddings.weight
        else:
            self.vocab_projection = nn.Linear(self.hidden_dim, self.vocab_size)

        self._reset_parameters()

    def _run_cell(self, input, md_layer, hidden, w_ih, w_hh, b_ih, b_hh):
        """
        LSTM cell structure adapted from:
        github.com/pytorch/benchmark/blob/09eaadc1d05ad442b1f0beb82babf875bbafb24b/rnns/fastrnns/cells.py#L25-L40
        """

        hx, cx = hidden
        gates = torch.matmul(input, w_ih.t()) + torch.matmul(hx, w_hh.t()) +\
                md_layer + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        if self.use_layernorm:
            ingate = self.layernorm(ingate)
            forgetgate = self.layernorm(forgetgate)
            cellgate = self.layernorm(cellgate)
            outgate = self.layernorm(outgate)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def _reset_parameters(self):
        """ Basic randomization of parameters"""
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_normal_(weight)
            else:
                torch.nn.init.zeros_(weight) # bias vector

    def forward(self, input):
        input_ids = input["input"].long()
        input_lens = input["text_len"]

        input_embs = self.embeddings(input_ids)
        max_batch_size = input_embs.size(0)
        seq_len = input_embs.size(1)

        # Assuming input is batch_first - permutting for sequence first
        input_embs = input_embs.permute(1, 0, 2)

        zeros = torch.zeros(self.n_layers, max_batch_size, self.hidden_dim).to(device)

        h_init = zeros
        c_init = zeros
        inputs = input_embs
        outputs = []

        md_input = self.metadata_constructor.preprocess_md(input["md"], self.embeddings)

        if self.metadata_constructor.is_precomputable():
            md = self.metadata_constructor(md_input, input_embs)

        for layer in range(self.n_layers):
            h = h_init[layer]
            c = c_init[layer]

            weight_start_index = layer * self._params_per_layer
            weight_end_index = (layer+1) * self._params_per_layer
            w_mh, w_ih, w_hh, b_ih, b_hh = self._all_weights[weight_start_index: weight_end_index]

            # Meta data can be computed in advance when not using attention

            if self.metadata_constructor.is_precomputable():
                precomputed_md = torch.matmul(md, w_mh.t())

            for t in range(seq_len):
                if not self.metadata_constructor.is_precomputable():
                    md = self.metadata_constructor(md_input, h)
                    md_layer = torch.matmul(md, w_mh.t())
                else:
                    md_layer = precomputed_md[t] if precomputed_md.dim() == 3 else precomputed_md

                h, c = self._run_cell(inputs[t], md_layer, (h, c), w_ih, w_hh, b_ih, b_hh)
                outputs += [h]

            inputs = outputs
            outputs = []

        # At the end the input variable will be set to outputs
        # Permutting to have batch - seq len - hidden dim
        lstm_out = torch.stack(inputs).permute(1, 0, 2)

        if self.use_weight_tying:
            vocab_predictions = self.vocab_projection(self.embedding_projection(lstm_out))
        else:
            vocab_predictions = self.vocab_projection(lstm_out)

        if self.use_softmax_adaptation:
            md_embs = md_embs.view(max_batch_size, -1)
            md_context = self.md_vocab_projection(md_embs).unsqueeze(1)
            vocab_predictions += md_context

        return vocab_predictions


class BaseLSTM(nn.Module):
    """Basic LSTM model - concatenating metadata to LSTM """
    def __init__(self, dimension_params, metadata_constructor_params, layer_params):
        super().__init__()
        self.emb_dim = dimension_params["emb_dim"]
        self.hidden_dim = dimension_params["hidden_dim"]
        self.vocab_size = dimension_params["vocab_size"]

        self.md_dims = metadata_constructor_params["md_dims"]
        self.md_group_sizes = metadata_constructor_params["md_group_sizes"]
        self.use_md = True if self.md_dims and self.md_group_sizes else False

        self.n_layers = layer_params["n_layers"]
        self.use_weight_tying = layer_params["use_weight_tying"]

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)

        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            )

        if self.use_weight_tying:
            self.vocab_projection = nn.Linear(self.emb_dim, self.vocab_size)
            self.embedding_projection = nn.Linear(self.hidden_dim, self.emb_dim)
            self.vocab_projection.weight = self.embeddings.weight
        else:
            self.vocab_projection = nn.Linear(self.hidden_dim, self.vocab_size)

        if self.use_md:
            self.metadata_constructor = MetadataConstructor(metadata_constructor_params,
                                                            dimension_params)

    def forward(self, input):
        """ Appends meta data embeddings to input and passes through LSTM."""
        input_ids = input["input"].long()
        input_lens = input["text_len"]
        input_embs = self.embeddings(input_ids)

        if self.use_md:
            md_input = self.metadata_constructor.preprocess_md(input["md"], self.embeddings)
            md_emb = self.metadata_constructor(md_input)

            # Prepending meta data information to text
            sos_embs = input_embs[:, :1, :]
            text_embs = input_embs[:, 1:, :]

            joined_embs = torch.cat((sos_embs, md_emb, text_embs), dim=1)
            joined_lens = 1 + input_lens # fixed metadata embedding

            packed_input = pack_padded_sequence(joined_embs, joined_lens,
                                                batch_first=True, enforce_sorted=False)
        else:
            packed_input = pack_padded_sequence(input_embs, input_lens,
                                                batch_first=True, enforce_sorted=False)

        # output: batch, seq_len, hidden_size
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        if self.use_weight_tying:
            vocab_predictions = self.vocab_projection(self.embedding_projection(lstm_out))
        else:
            vocab_predictions = self.vocab_projection(lstm_out)

        return vocab_predictions

##### Utility Wrapper  #####

def init_base_lstm(model_config, vocab_size):

    dimension_params = {
        "emb_dim":int(model_config.get("emb_dim")),
        "context_dim":int(model_config.get("emb_dim")),
        "hidden_dim":int(model_config.get("hidden_dim")),
        "vocab_size":vocab_size,
    }

    metadata_constructor_params = {
        "md_projection_dim":int(model_config.get("md_projection_dim", fallback="50")),
        "md_dims":[int(x) for x in model_config.get("md_dims", fallback="").split(',') if x],
        "md_group_sizes":[int(x) for x in model_config.get("md_group_sizes", fallback="").split(',') if x],
        "attention_mechanism": "",
        "hierarchical_attention": "",
    }

    layer_params = {
        "n_layers":1, # Fixed in ACL paper
        "use_weight_tying":eval(model_config.get("use_weight_tying", fallback='False')),
    }


    model = BaseLSTM(dimension_params, metadata_constructor_params, layer_params)
    return model


def init_concat_lstm(model_config, vocab_size):
    dimension_params = {
        "emb_dim":int(model_config.get("emb_dim")),
        "context_dim":int(model_config.get("context_dim",
                      fallback=int(model_config.get("emb_dim")))),
        "hidden_dim":int(model_config.get("hidden_dim")),
        "vocab_size":vocab_size,
    }

    metadata_constructor_params = {
        "md_projection_dim":int(model_config.get("md_projection_dim", fallback="50")),
        "md_dims":[int(x) for x in model_config.get("md_dims").split(',')],
        "md_group_sizes":[int(x) for x in model_config.get("md_group_sizes").split(',')],
        "attention_mechanism":model_config.get("attention_mechanism", fallback=""),
        "query_type":model_config.get("query_type", fallback=""),
        "use_null_token":eval(model_config.get("use_null_token", fallback="False")),
        "hierarchical_attention":eval(model_config.get("hierarchical_attention", fallback="False")),
    }

    layer_params = {
        "n_layers":1, # Fixed in ACL paper
        "use_softmax_adaptation":eval(model_config.get("use_softmax_adaptation", fallback='False')),
        "use_layernorm":eval(model_config.get("use_layernorm", fallback='False')),
        "use_weight_tying":eval(model_config.get("use_weight_tying", fallback='False')),
    }

    model = ConcatLSTM(dimension_params, metadata_constructor_params, layer_params)

    return model


MODEL_MAP = {"base_lstm": init_base_lstm,
             "concat_lstm": init_concat_lstm}


def get_model(config, vocab_size):
    """Given a model type maps that to a model initiation function."""
    model_config = config["MODEL"]
    model_type = model_config.get("model_type")
    assert(model_type in MODEL_MAP), f"Invalid model type: {model_type}"
    return MODEL_MAP[model_type](model_config, vocab_size)
