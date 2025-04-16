import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from .attention_mask import make_mask
from .configuration_vora import VoRAConfig
from .vision_embedding import *  # hacking, let transformers find vision_embedding
from . import vision_embedding as VB
from .lora import apply_lora
from .vora_generation_utils import (
    VoraGenerationMixin,
    custom_prepare_4d_causal_attention_mask_with_cache_position,
)

try:
    from utils import logging
except:
    from transformers.utils import logging


logger = logging.get_logger(__name__)


class VoRAForCausalLM(PreTrainedModel):
    config_class = VoRAConfig
    _auto_class = 'AutoModelForCausalLM'
    supports_gradient_checkpointing = True
    supports_report_metrics: bool = True

    def __init__(self, config: PretrainedConfig = VoRAConfig()):
        super().__init__(config)
        self.config = config
        # -------------- Setup LLM ---------------------
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm)
        self.llm.__class__ = type(self.llm.__class__.__name__, (self.llm.__class__, VoraGenerationMixin), {})
        self.llm.model._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(custom_prepare_4d_causal_attention_mask_with_cache_position)

        self.config.update(self.llm.config.to_dict())

        # -------------- Setup LoRA -------------------
        if config.lora:
            for _, param in self.llm.named_parameters():
                param.requires_grad = False
            apply_lora(self.llm, config.lora)
        # ----------------------------------------------

        # ------------ Setup Vision Embedding ----------
        self.vision_embedding = getattr(VB, config.vision_embedding)(self.config)  # setup after llm so that we know the hiddensize
        # ----------------------------------------------

        # ------------- Setup Aux Vision ---------------
        self.enable_aux_vision = False
        if config.aux_vision:
            from .aux_vision import AuxVision
            self.enable_aux_vision = True
            self.aux_vision = AuxVision(self.config)
            if config.reuse_aux_vision_embedding_layers:
                weights = getattr(self.aux_vision.aux_model, config.reuse_aux_vision_embedding_layers).state_dict()
                msg = self.vision_embedding.load_state_dict(weights, strict=False)
                logger.info(f"Loaded aux vision weights: {msg}")
        # ----------------------------------------------
        # print trainable prameters and total parameters so that we can check if we are loading the correct model
        logger.info("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}: {param.numel()}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters())}")

    def detach_and_gather_loss(self, loss, dtype, device):
        if not dist.is_initialized():
            return loss.item()
        gathered_loss = [torch.tensor(0.0, dtype=loss.dtype).to(device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_loss, loss.detach().clone())
        avg_gathered_loss = torch.mean(torch.stack(gathered_loss))
        return avg_gathered_loss.item()

    def _encode_vision(self, images, n_frames):
        # TODO: we need a more elegant way here to deal with mixed image and pure text training
        if images.size(0) > 0:
            vision_embeds = self.vision_embedding(images)
        else:
            # FIXME: hacking for deepspeed training
            # we feed a dummy image tensor (1, 3, H, W) into vision_encoder when training a pure-text batch
            images = images.new_zeros((1, *images.shape[1:]))
            vision_embeds = self.vision_embedding(images)[0:0]
        vision_embeds = vision_embeds.split(n_frames, dim=0)
        attention_mask = [torch.ones(feature.size()[:-1], dtype=torch.long).to(feature.device) for feature in vision_embeds]
        vision_targets = [torch.ones(feature.size(), dtype=torch.long).to(feature.device).fill_(-100) for feature in attention_mask]

        image_shapes = images.shape[-2:]

        return vision_embeds, attention_mask, vision_targets, image_shapes

    def _concat_embedding(self, vision_encode_out, batch, vision_placeholder_index, left_padding=False):
        """ concat vision and text
        """

        vision_embeds, vision_atts, vision_targets, _ = vision_encode_out

        input_embeds = []
        attention_mask = []
        targets = []
        vision_mask = []  # set vision token as 1, text token as 0

        for cur_batch_idx, cur_input_ids in enumerate(batch["input_ids"]):
            cur_vision_embeds = vision_embeds[cur_batch_idx]
            cur_vision_attn = vision_atts[cur_batch_idx]
            cur_vision_targets = vision_targets[cur_batch_idx]
            cur_attn_masks = batch["attention_mask"][cur_batch_idx]

            image_token_indices = torch.where(cur_input_ids == vision_placeholder_index)[0]
            cur_image_num = len(image_token_indices)
            image_token_indices = list(image_token_indices) + [cur_input_ids.shape[0]]

            cur_input_embeds = []
            cur_attention_mask = []
            cur_target = []
            cur_vision_mask = []

            # convert text before 1st <image> to embedding
            image_token_index = image_token_indices[0]

            cur_input_embeds.append(
                self.llm.get_input_embeddings()(cur_input_ids[:image_token_index]),
            )
            cur_attention_mask.append(
                cur_attn_masks[:image_token_index],
            )
            cur_vision_mask.append(
                torch.zeros_like(cur_attn_masks[:image_token_index]).to(cur_attn_masks.device),
            )
            if "labels" in batch:
                cur_target.append(
                    batch["labels"][cur_batch_idx, :image_token_index],
                )

            if batch.get("vison_placeholder_mode", 0) == 1:
                assert cur_image_num <= 1, "multiple video input is not supported"
                cur_vision_embeds = cur_vision_embeds.unsqueeze(0)
                cur_vision_attn = cur_vision_attn.unsqueeze(0)
                cur_vision_targets = cur_vision_targets.unsqueeze(0)
            assert cur_image_num == len(cur_vision_embeds), \
                f"Size mismatch! cur_image_num: {cur_image_num}, len(cur_vision_embeds): {len(cur_vision_embeds)} {len(cur_vision_embeds)} \
                    in {batch['prompt'][cur_batch_idx]} & {batch['gt'][cur_batch_idx]} & {batch['input_ids'][cur_batch_idx]}"
            # convert each <image> xxx group into embedding
            text_embedding = self.llm.get_input_embeddings()(cur_input_ids.relu())
            for i in range(0, cur_image_num):
                image_token_index = image_token_indices[i]
                cur_input_embeds.extend([
                    cur_vision_embeds[i],
                    text_embedding[image_token_index+1:image_token_indices[i+1]]
                ])
                cur_attention_mask.extend([
                    cur_vision_attn[i],
                    cur_attn_masks[image_token_index+1:image_token_indices[i+1]]
                ])
                cur_vision_mask.extend([
                    torch.ones_like(cur_vision_attn[i]).to(cur_vision_attn[i].device),
                    torch.zeros_like(cur_attn_masks[image_token_index+1:image_token_indices[i+1]]).to(cur_vision_attn[i].device),
                ])
                if "labels" in batch:
                    cur_target.extend([
                        cur_vision_targets[i],
                        batch["labels"][cur_batch_idx, image_token_index+1:image_token_indices[i+1]],
                    ])

            input_embeds.append(torch.cat(cur_input_embeds))
            attention_mask.append(torch.cat(cur_attention_mask))
            vision_mask.append(torch.cat(cur_vision_mask))
            if "labels" in batch:
                targets.append(torch.cat(cur_target))

        # padding
        n_tokens = [embed.shape[0] for embed in input_embeds]

        max_token = max(n_tokens)

        for i in range(len(input_embeds)):
            if max_token > n_tokens[i]:
                self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                pad_token = torch.tensor([self.pad_id] * (max_token - n_tokens[i]))
                pad_embedding = self.llm.get_input_embeddings()(pad_token.to(batch["attention_mask"][i].device))
                pad_attention = torch.zeros(pad_embedding.shape[0], dtype=torch.long).to(batch["attention_mask"][i].device)
                pad_targets = torch.ones(pad_attention.size(), dtype=torch.long).to(batch["attention_mask"][i].device).fill_(-100)

                if left_padding:
                    input_embeds[i] = torch.cat([pad_embedding, input_embeds[i]])
                    attention_mask[i] = torch.cat([pad_attention, attention_mask[i]])
                    vision_mask[i] = torch.cat([pad_attention, vision_mask[i]])
                    if "labels" in batch:
                        targets[i] = torch.cat([pad_targets, targets[i]])
                else:
                    input_embeds[i] = torch.cat([input_embeds[i], pad_embedding])
                    attention_mask[i] = torch.cat([attention_mask[i], pad_attention])
                    vision_mask[i] = torch.cat([vision_mask[i], pad_attention])
                    if "labels" in batch:
                        targets[i] = torch.cat([targets[i], pad_targets])

        inputs_embeds = torch.stack(input_embeds, dim=0).type(self.llm.dtype)
        attention_mask = torch.stack(attention_mask, dim=0)
        vision_mask = torch.stack(vision_mask, dim=0).to(attention_mask.device)

        if len(targets) > 0:
            targets = torch.stack(targets, dim=0)

        attention_mask = make_mask(
            attention_mask,
            mode=self.config.vision_attention_mask,
            vision_mask=vision_mask,
            dtype=inputs_embeds.dtype
        )

        return inputs_embeds, attention_mask, targets, vision_mask

    def forward(self, **batch):
        # -------------- Vision/Text Embedding ----------
        vision_placeholder_index = batch.pop("vision_placeholder_index")
        images, n_frames = batch["frames"], batch["n_frames"]
        vision_encode_out = self._encode_vision(images, n_frames)
        inputs_embeds, attention_mask, targets, vision_mask = self._concat_embedding(
            vision_encode_out, batch, vision_placeholder_index)
        # -----------------------------------------------

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True,
            output_hidden_states=True,
        )

        llm_loss = outputs.loss
        device = llm_loss.device
        dtype = llm_loss.dtype

        metrics = {}

        metrics["llm_loss"] = self.detach_and_gather_loss(llm_loss, dtype, device)
        if self.enable_aux_vision:
            if images.size(0) > 0:
                aux_losses = self.aux_vision(images, outputs.hidden_states, vision_mask)
            else:
                # FIXME: hacking for deepspeed training
                aux_losses = {key: torch.tensor(0., dtype=dtype).to(device) for key in self.aux_vision.loss_keys}

            aux_loss = torch.tensor(0., dtype=dtype).to(device)
            n_aux = 0
            for _aux_key, _aux_loss in aux_losses.items():
                aux_loss += _aux_loss
                n_aux += 1
                metrics[_aux_key] = self.detach_and_gather_loss(_aux_loss, dtype, device)
            aux_loss /= n_aux

            outputs.loss = aux_loss + llm_loss
        metrics["total_loss"] = self.detach_and_gather_loss(outputs.loss, dtype, device)
        self.report_metrics(**metrics)

        return outputs

    def generate(self, batch, **generate_params):

        with torch.amp.autocast(
            enabled=(self.device != torch.device("cpu")),
            device_type=self.device.type,
        ):
            # get vision token
            vision_placeholder_index = batch.pop("vision_placeholder_index")

            # get vision features
            images, n_frames = batch["frames"], batch["n_frames"]
            vision_encode_out = self._encode_vision(images, n_frames)

            inputs_embeds, attention_mask, _, _ = self._concat_embedding(
                vision_encode_out, batch, vision_placeholder_index, left_padding=False)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            **generate_params
        )

        return outputs
