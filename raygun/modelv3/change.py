Copyright © 2025 Kavi Shah

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import reduce, repeat, rearrange
import numpy as np
import os
import math
from esm.model.esm2 import TransformerLayer
from einops.layers.torch import Rearrange





class ReductionAndExpansionAreaResamp(nn.Module):
    """
    Resamples sequences with variable input and output lengths using 'area' interpolation.
    Supports masking of padded input tokens and variable target lengths per batch.

    Args:
        None (can later add stdev/noise parameters)

    Inputs:
        x: Tensor of shape [B, L_max, D]
        padding_mask: Bool tensor [B, L_max], where True = padded (invalid)
        finallength: int | list[int] | tensor[int]
            Target length(s) per batch element.

    Outputs:
        padded_out: Tensor [B, L_out_max, D] (zero-padded)
        out_mask:   Bool tensor [B, L_out_max] (True = padded)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        finallength,
        padding_mask: torch.Tensor = None,
    ):
        B, L_max, D = x.shape

        if padding_mask is None:
            padding_mask = torch.zeros(B, L_max, dtype=torch.bool, device=x.device)
        
        assert padding_mask.shape == (B, L_max), "padding_mask must be [B, L_max]"

        # Normalize finallength → Tensor[B]
        if isinstance(finallength, int):
            finallengths = torch.full((B,), finallength, dtype=torch.long, device=x.device)
        else:
            finallengths = torch.as_tensor(finallength, dtype=torch.long, device=x.device)
            assert finallengths.shape[0] == B, "finallength must have one value per batch element"

        # --- Preallocate outputs ---
        max_len_out = finallengths.max().item()
        padded_out = torch.zeros(B, max_len_out, D, dtype=x.dtype, device=x.device)
        out_mask = torch.ones(B, max_len_out, dtype=torch.bool, device=x.device)  # all padded initially

        # --- Single pass loop with enumerate ---
        for b, target_len in enumerate(finallengths.tolist()):
            # Extract valid (unpadded) region
            valid_len = (~padding_mask[b]).sum().item()
            seq = x[b, :valid_len]  # [L_b, D]

            # Interpolate to target length
            seq = seq.unsqueeze(0).transpose(1, 2).unsqueeze(1)  # [1, 1, D, L_b]
            out = F.interpolate(seq, size=(D, target_len), mode="area")
            out = out.squeeze(1).transpose(1, 2)  # [1, target_len, D]

            # Place in padded_out and mark valid positions in mask
            Lb = out.shape[1]
            padded_out[b, :Lb] = out
            out_mask[b, :Lb] = False  # False = valid, True = padded

        return padded_out, out_mask
