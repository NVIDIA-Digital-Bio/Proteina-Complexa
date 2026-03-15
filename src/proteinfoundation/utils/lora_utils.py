# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


import loralib as lora
from torch import nn


def replace_lora_layers(module, r, lora_alpha, lora_dropout, exclude_keys=()):
    """
    Recursively replace all nn.Linear and nn.Embedding layers in the module with the lora.Linear and lora.Embedding layers.

    Args:
        module (nn.Module): The module containing nn.Linear layers to be replaced.
        r (int): The rank for the pair of low-rank adaptation matrices.
        lora_alpha (float): Used for calculating lora scaling.
        lora_dropout (float): Dropout rate for the input before LoRA layers.
        exclude_keys (tuple): Keys to exclude from replacement.
    """
    for name, child in module.named_children():
        if name == "pair_update" or any(exclude_key in name for exclude_key in exclude_keys):
            print(f"Skipping {name} because it is in exclude_keys")
            continue
        # Check if the child module is an instance of nn.Linear
        if isinstance(child, nn.Linear):
            # Replace the nn.Linear layer with a new one
            if not child.weight.requires_grad:
                continue
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            # Turn off merge weights, as turning it on behaves strangely in the training mode of lightning trainer
            new_layer = lora.Linear(
                in_features,
                out_features,
                r,
                lora_alpha,
                lora_dropout,
                merge_weights=False,
                bias=bias,
            )
            setattr(module, name, new_layer)
        elif isinstance(child, nn.Embedding):
            # Replace nn.Embedding layer
            if not child.weight.requires_grad:
                continue
            num_embeddings = child.num_embeddings
            embedding_dim = child.embedding_dim
            padding_idx = child.padding_idx
            max_norm = child.max_norm
            norm_type = child.norm_type
            scale_grad_by_freq = child.scale_grad_by_freq
            sparse = child.sparse
            new_layer = lora.Embedding(
                num_embeddings,
                embedding_dim,
                r,
                lora_alpha,
                merge_weights=False,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
            )
            setattr(module, name, new_layer)
        else:
            # Recursively replace layers in submodules
            replace_lora_layers(child, r, lora_alpha, lora_dropout)
