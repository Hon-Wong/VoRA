from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["VoRAConfig"]


class VoRAConfig(PretrainedConfig):
    model_type = "vora"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        llm: str = "",
        aux_vision: str = "",
        skip_aux_cls: bool = False,
        reuse_aux_vision_embedding_layers: str = "",
        lora: dict = {}, 
        image_size: int = 448,
        vision_embedding: str = "AIMv2",
        vision_embedding_intermediate_size: int = 1536,
        patch_size: int = 14,
        vision_attention_mask: str = "bidirectional",
        rms_norm_eps: float = 1e-5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.aux_vision = aux_vision
        self.skip_aux_cls = skip_aux_cls
        self.reuse_aux_vision_embedding_layers = reuse_aux_vision_embedding_layers
        self.lora = lora
        self.image_size = image_size
        self.vision_embedding = vision_embedding
        self.vision_embedding_intermediate_size = vision_embedding_intermediate_size
        self.patch_size = patch_size
        self.vision_attention_mask = vision_attention_mask
        self.rms_norm_eps = rms_norm_eps
    