import torch
import torch.nn as nn

from .configuration_vora import VoRAConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: VoRAConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            config.vision_embedding_intermediate_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.vision_embedding_intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AIMv2Embedding(nn.Module):
    def __init__(self,
        config: VoRAConfig = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        num_patches = (config.image_size // config.patch_size) ** 2
        self.config = config

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.vision_embedding_intermediate_size)))
        self.out_proj = nn.Linear(config.vision_embedding_intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_token = H // self.config.patch_size
        w_token = W // self.config.patch_size
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)

        if N <= pos_embed.size(1):
            tokens = tokens + pos_embed[:, :N]
        else:
            pos_embed = pos_embed.view(1, int(pos_embed.size(1)**0.5), int(pos_embed.size(1)**0.5), -1).permute(0, 3, 1, 2)
            pos_embed = nn.functional.interpolate(pos_embed, size=(h_token, w_token), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            pos_embed = pos_embed.view(1, N, pos_embed.size(-1))
            tokens = tokens + pos_embed

        return self.out_proj(tokens)
