from typing import Optional

import torch


def _make_causal_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = attention_mask.shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def _make_2dvison_mask(column_mask, dtype: torch.dtype, device: torch.device):
    """
    """
    bsz, seq_length = column_mask.shape
    cross_mask = torch.zeros((bsz, 1, seq_length, seq_length), dtype=dtype, device=device)

    # 找到连续的 1 的区间
    start = None
    for bsz_idx in range(bsz):
        for i in range(seq_length):
            if column_mask[bsz_idx, i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # 填充区间
                    cross_mask[bsz_idx, 0, start:i, start:i] = 1
                    start = None

        # 处理最后一个区间
        if start is not None:
            cross_mask[bsz_idx, 0, start:seq_length, start:seq_length] = 1

    return cross_mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill_(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def make_mask(attention_mask: torch.Tensor, dtype: torch.dtype=None, device: torch.device=None, mode: str="default", vision_mask: torch.Tensor=None, ):
    if dtype is None:
        dtype = attention_mask.dtype
    if device is None:
        device = attention_mask.device      
    expanded_attn_mask = _expand_mask(attention_mask, dtype).to(device)
    causal_mask = _make_causal_mask(attention_mask, dtype, device).to(device)
    if mode == "default":
        return attention_mask
    else:
        assert vision_mask is not None, "vision_mask is None"
        vision_mask = vision_mask.to(device)
        bsz, seq_length = attention_mask.shape
        vision_mask_bg = vision_mask[:, None, :, None]
        vision_mask_2d = _make_2dvison_mask(vision_mask, dtype, device)
        if mode == "bidirectional":
            mask = expanded_attn_mask + causal_mask
            mask = mask.clone().masked_fill_(vision_mask_2d.to(torch.bool), 0)
            return mask 
        else:
            raise NotImplementedError(f"mode {mode} is not implemented")
