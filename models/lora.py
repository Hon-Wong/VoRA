import torch
import types
import math
from torch import nn
import torch.nn.functional as F


QWEN2_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]


class LoRALayer(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        r: int = 1024,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features)
        # we elimate lora_alpha here bc we find it unnecessary in VoRA
        if r < 0:
            self.forward = self.naive_forward
        else:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor):
        intermediate = F.linear(x, self.weight, bias=self.bias)
        result = intermediate + self.lora_B(self.lora_A(x))
        return result

    def naive_forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, bias=self.bias)

def _get_submodules(self, key):
    parent = self.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = self.get_submodule(key)
    return parent, target, target_name

def _find_and_replace(self, lora_params):
    target_modules = lora_params["target_modules"]

    for llm_module_name in target_modules:
        parent, target, target_name = self._get_submodules(llm_module_name)
        bias = target.bias is not None
        vora_layer = LoRALayer(
            target.in_features,
            target.out_features,
            bias=bias,
            **lora_params
        )
        self._replace_module(parent, target_name, vora_layer, target)

def _replace_module(self, parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if old_module.bias is not None:
        new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

def apply_lora(llm, lora_params={"layers": "all", "r": 1024, "target_modules": QWEN2_TARGET_MODULES}):
    llm_num_layers = llm.config.num_hidden_layers
    total_layers = lora_params.get("layers", "all")

    # -------------------- validation check ---------------------
    if isinstance(total_layers, str):
        if total_layers.lower() == "all":
            total_layers = list(range(llm_num_layers))
    else:
        assert isinstance(total_layers, int), "total_layers must be an integer or 'all'"
        total_layers = list(range(total_layers))
    # -------------------- validation check ---------------------

    # -------------------- replace llm layers ---------------------
    for i in total_layers:
        llm_layer = llm.model.layers[i]
        llm_layer._get_submodules = types.MethodType(_get_submodules, llm_layer)
        llm_layer._find_and_replace = types.MethodType(_find_and_replace, llm_layer)
        llm_layer._replace_module = types.MethodType(_replace_module, llm_layer)
        llm_layer._find_and_replace(lora_params)

if __name__ == "__main__":
    from transformers import LlamaForCausalLM, CLIPVisionModel, AutoModel
    llama = LlamaForCausalLM.from_pretrained("/mnt/bn/wh-data/data/models/llama2_7b_hf_chat")
    apply_lora(llama)
    print(llama)