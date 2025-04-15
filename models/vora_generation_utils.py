from typing import Any, Dict, Optional

import torch
from transformers import GenerationMixin
from transformers.cache_utils import Cache
from transformers.utils import ModelOutput


class VoraGenerationMixin(GenerationMixin):

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask_2d = (attention_mask[:, 0, :, :] == 0).any(dim=1).long().to(attention_mask.device)
            model_input = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask_2d,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )
            model_input['attention_mask'] = attention_mask
            return model_input
        else:
            return super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"].ndim == 4:
            attention_mask = model_kwargs.pop("attention_mask")
            model_kwargs = super()._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
            )
            bs, _, seq_len, tgt_len = attention_mask.shape
            dtype = attention_mask.dtype
            min_dtype = torch.finfo(dtype).min
            new_col = attention_mask.new_zeros((bs, 1, seq_len, 1)).fill_(min_dtype)
            new_row = attention_mask.new_zeros((bs, 1, 1, tgt_len + 1))
            model_kwargs["attention_mask"] = torch.cat([
                torch.cat([attention_mask, new_col], dim=-1),
                new_row
            ], dim=2)
            return model_kwargs
        else:
            return super()._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
            )


def custom_prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask[:, :, -sequence_length:, -target_length:]
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask
