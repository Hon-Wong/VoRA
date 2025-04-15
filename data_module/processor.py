from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import os
import random
from typing import Any, List, Union

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from data_module.vision_transform import VisionProcessor
from data_module.vqa_processor import VQAProcessor
from utils import logging
from utils.constants import (
    DATA_TYPE_IMAGE,
    DATA_TYPE_TEXT,
    DATA_TYPE_VIDEO,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


logger = logging.get_logger(__name__)


def load_local_frame(frame_path):
    return Image.open(frame_path).convert("RGB")


def random_index(frames_length, num_segments, average=False):
    if frames_length <= num_segments:
        return [i for i in range(frames_length)] + (num_segments - frames_length) * [frames_length - 1]
    else:
        result = []
        stride = frames_length // num_segments
        s_list = [stride] * num_segments
        for i in range(frames_length - num_segments * stride):
            s_list[i] += 1
        if not average:
            random.shuffle(s_list)
        cursor = 0
        for each_stride in s_list:
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if not average:
                result.append(random.randint(left, right - 1))
            else:
                result.append(left)
        return result


class VoRAProcessor(object):
    def __init__(self,
                 frames_key: str = "frames",
                 frames_ops: Any = None,
                 aux_frames_ops: Any = None,
                 label_key: Union[str, List, None] = None,
                 meta_keys: List[str] = ["id", "source", "question", "gt"],
                 padding_side: str = "right",
                 tokenizer: str = "",
                 trust_remote_code: bool = False,
                 eos_token: Union[str, None] = None,
                 max_seq_len: int = 512,
                 max_prompt_len: int = 512,
                 sample_method: str = "global_random",
                 dummy_frame_shape = (0, 3, 448, 448),
                 max_batch_frames: int = 16,
                 num_segments: int = 8,
                 training: bool = True,
                 verbose: bool = True,
                 task_type: str = "completion",
                 truncate_mode: str = "qa",
                 vqa_processor_params: dict = {},
                 ):
        self.frames_key = frames_key
        self.num_segments = num_segments
        self.max_batch_frames = max_batch_frames
        self.truncate_mode = truncate_mode
        self.dummy_frame_shape = dummy_frame_shape
        self.label_key = label_key
        self.meta_keys = meta_keys
        self.padding_side = padding_side
        self.training = training
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.sample_method = sample_method
        self.verbose = verbose
        self.aux_frames_ops = aux_frames_ops
        self.enable_aux_frames = False

        # vision processors
        if isinstance(frames_ops, str):
            self.video_processor = AutoImageProcessor.from_pretrained(
                frames_ops)
        else:
            self.video_processor = VisionProcessor(frames_ops)
        self.enable_aux_frames = (aux_frames_ops is not None)
        if self.enable_aux_frames:
            assert isinstance(aux_frames_ops, dict), "aux_frames_ops must be dict"
            self.aux_frame_keys = list(aux_frames_ops.keys())
            self.aux_video_processor = {}
            for name in self.aux_frame_keys:
                current_aux_frames_ops = aux_frames_ops[name]
                if isinstance(current_aux_frames_ops, str):
                    self.aux_video_processor[name] = AutoImageProcessor.from_pretrained(
                        current_aux_frames_ops)
                else:
                    self.aux_video_processor[name] = VisionProcessor(current_aux_frames_ops)

        # load tokenizer
        self.eos_token = eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=False, trust_remote_code=trust_remote_code)
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.convert_tokens_to_ids(eos_token) \
            if eos_token else self.tokenizer.eos_token_id
        self.ignore_index = IGNORE_INDEX

        self.task_type = task_type
        self.vision_placeholder = DEFAULT_IMAGE_TOKEN
        self.vision_placeholder_index = IMAGE_TOKEN_INDEX

        self.vqa_processor = VQAProcessor(
            self.label_key, self.vision_placeholder, **vqa_processor_params)

    def preprocess(self, data_dict):
        has_frame = False
        if self.frames_key in data_dict:
            has_frame = True
            data_dict["data_type"] = DATA_TYPE_IMAGE
            if len(data_dict[self.frames_key]) > 1:
                data_dict["data_type"] = DATA_TYPE_VIDEO

            self.sample_frames(data_dict)
            frames = data_dict[self.frames_key]
            if isinstance(frames[0], (str, os.PathLike)):
                frames = [os.path.join(
                    data_dict["image_folder"], frame) for frame in frames]
                with ThreadPoolExecutor(max_workers=32) as executor:
                    frames = [frame for frame in executor.map(
                        load_local_frame, frames)]
                data_dict["frame_paths"] = frames
            elif isinstance(frames[0], bytes):
                frames = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in frames]
            num_frames = len(frames)
            data_dict["n_frames"] = num_frames
            data_dict[self.frames_key] = frames
        else:
            data_dict[self.frames_key] = torch.zeros(self.dummy_frame_shape)
            data_dict["n_frames"] = 0
            data_dict["data_type"] = DATA_TYPE_TEXT
        
        if self.label_key not in data_dict:
            logger.error(f"label_key {self.label_key} not in data_dict")
            raise ValueError(f"label_key {self.label_key} not in data_dict")
        else:
            if isinstance(data_dict[self.label_key][0], list):
                data_dict[self.label_key] = data_dict[self.label_key][0]
        
        if has_frame and self.frames_key in data_dict:
            self.add_vision_placeholders_in_prompt(data_dict)
        data_dict["has_frame"] = has_frame
        return data_dict

    def sample_frames(self, data_dict):
        average_draw = not self.training
        frames = data_dict[self.frames_key]
        if self.sample_method == "global_random":
            frames_index = random_index(
                len(frames), self.num_segments, average_draw)
            part_frames = [frames[i] for i in frames_index]
        elif self.sample_method == "global":
            part_frames = frames
            if len(part_frames) > self.max_batch_frames:
                frames_index = random_index(
                    len(part_frames), self.max_batch_frames, average_draw)
                part_frames = [part_frames[i] for i in frames_index]
        else:
            raise NotImplementedError(f"sample method {self.sample_method} not implemented")
        data_dict[self.frames_key] = part_frames
        return

    def add_vision_placeholders_in_prompt(self, data_dict):
        """ For mixture training with video/image datasets, we refine media tokens in prompt.
            - in image mode: replace <video> with [Frame i: <image>] * n_frames
            - in video mode: replace <image> with <video> directly
        """
        def _add_timestamp(frame_count, frame_prefix_pattern="{i}s: ", offset=1, sep="; ", end_symbol="\n"):
            if frame_count == 1:
                return DEFAULT_IMAGE_TOKEN
            image_mode_prompt = ""
            for i in range(frame_count):
                if "{i}" in frame_prefix_pattern:
                    frame_prefix = frame_prefix_pattern.format(i=i+offset)
                else:
                    frame_prefix = frame_prefix_pattern
                image_mode_prompt += frame_prefix + DEFAULT_IMAGE_TOKEN + sep
            return image_mode_prompt + end_symbol

        # in image mode, replace <video> with [Frame i: <image>] * n_frames
        if self.label_key in data_dict:
            image_mode_prompt = _add_timestamp(len(data_dict[self.frames_key]))
            vision_token_exist = False
            for item in data_dict[self.label_key]:
                if DEFAULT_VIDEO_TOKEN in item["value"]:
                    vision_token_exist = True
                    item["value"] = item["value"].replace(
                        DEFAULT_VIDEO_TOKEN, image_mode_prompt)
                elif DEFAULT_IMAGE_TOKEN in item["value"]:
                    vision_token_exist = True
                    break
            if not vision_token_exist:
                # add vision token to the beginning of the prompt
                data_dict[self.label_key][0]["value"] = image_mode_prompt \
                    + data_dict[self.label_key][0]["value"]

    def transform(self, data_dict):
        try:
            data_dict = self.preprocess(data_dict)

            output = dict()
            # add meta data
            for key in self.meta_keys:
                output[key] = data_dict.get(key, "unknown")

            # add vision info
            output["n_frames"] = data_dict["n_frames"]
            output["data_type"] = data_dict["data_type"]
            if self.frames_key in data_dict:
                output.update(self.build_visual(data_dict))

            output.update(self.build_text(data_dict))
            return output
        except Exception as e:
            logger.warning(f"Collaped data!!! {e}")
            return None
    
    def batch_process(self, data_dict):
        keys = list(data_dict.keys())
        bs = len(data_dict[keys[0]])

        output = defaultdict(list)
        for i in range(bs):
            _data_dict = {key: data_dict[key][i] for key in keys}
            _output = self.transform(_data_dict)
            # FIXME: 这里可能会有问题，因为batch_size可能会不一致，后面如果用不到这个函数可以删除
            if _output is None:
                continue
            for key, value in _output.items():
                output[key].append(value)
        return output

    def tokenizer_vision_placeholder(self, prompt, has_frame, add_bos=False):
        def join_lists(*lists, sep):
            result = []
            for i, lst in enumerate(lists):
                if i > 0 and sep:
                    result.extend([sep])
                result.extend(lst)
            return result

        if has_frame:
            # We encounter cases where there is no image but <image> in prompt
            prompt_chunks = [self.tokenizer.encode(
                chunk) for chunk in prompt.split(self.vision_placeholder)]
            input_ids = join_lists(*prompt_chunks, sep=self.vision_placeholder_index)
        else:
            input_ids = self.tokenizer.encode(prompt)
        if add_bos:
            input_ids = [self.tokenizer.bos_token_id] + input_ids

        return input_ids

    def build_text(self, data_dict):
        prompt_list, response_list = self.vqa_processor(data_dict)
        has_frame = data_dict["has_frame"]
        prompt_token_ids_list = [self.tokenizer_vision_placeholder(prompt, has_frame) for prompt in prompt_list]
        response_token_ids_list = [self.tokenizer.encode(response) for response in response_list]
        input_ids = []
        label_mask = []

        for i, (prompt_id, response_id) in enumerate(zip(prompt_token_ids_list, response_token_ids_list)):
            total_length = len(input_ids) + len(prompt_id) + len(response_id)
            if total_length >= self.max_seq_len - 1 and self.truncate_mode == "qa" and i > 0:
                # truncating conversation instead of truncating a sentence
                logger.warning_rank0(f"Warning! Get incoming text length {total_length} >= max length {self.max_seq_len}. Truncate qa now.")
                break
            elif total_length >= self.max_seq_len - 1 and (self.truncate_mode == "text" or i == 0):
                logger.warning_rank0(f"Warning! Get incoming text length {total_length} >= max length {self.max_seq_len}. Truncate text now.")

                input_ids += prompt_id + response_id
                input_ids = input_ids[:self.max_seq_len - 1]
                label_mask += [0] * len(prompt_id) + [1] * len(response_id)
                label_mask = label_mask[:self.max_seq_len - 1]
                input_ids = input_ids + [self.eos_id]
                label_mask = label_mask + [1]
                break
            input_ids += prompt_id + response_id
            label_mask += [0] * len(prompt_id) + [1] * len(response_id)
            input_ids = input_ids + [self.eos_id]
            label_mask = label_mask + [1]
        
        input_mask = [1] * len(input_ids)
        prompt = " ".join(prompt_list)
        response = " ".join(response_list)

        input_ids = torch.as_tensor(input_ids, dtype=torch.int64)
        label_mask = torch.as_tensor(label_mask, dtype=torch.int64)
        attention_mask = torch.as_tensor(input_mask, dtype=torch.int64)
        label = input_ids.masked_fill(label_mask != 1, self.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "prompt": prompt,
            "gt": response
        }

    def padding_sequence(self, inputs, value):
        """ Pad input sequence(input_ids, attention_mask, label) to `max_length`,
            fill padding place with `value`
        """
        max_length = max([len(d) for d in inputs])
        padded_data = []
        for t in inputs:
            if len(t) < max_length:
                pad_len = max_length - len(t)
                pad = (0, pad_len) if self.padding_side == "right" else (pad_len, 0)
                t = F.pad(t, pad, value=value)
            padded_data.append(t)
        return torch.stack(padded_data)

    def build_visual(self, data_dict):
        frames = data_dict[self.frames_key]
        if data_dict["n_frames"] > 0:
            ret_dict = {}
            if isinstance(self.video_processor, VisionProcessor):
                ret = self.video_processor(frames)
                ret = torch.stack(ret)
            elif isinstance(self.video_processor, BaseImageProcessor):
                ret = self.video_processor([np.asarray(frame.convert(
                    'RGB')) for frame in frames], return_tensors="pt").data["pixel_values"]
            else:
                raise NotImplementedError
            ret_dict[self.frames_key] = ret
            if self.enable_aux_frames:
                for aux_frame_key in self.aux_frame_keys:
                    aux_video_processor = self.aux_video_processor[aux_frame_key]
                    if isinstance(aux_video_processor, VisionProcessor):
                        aux_ret = aux_video_processor(frames)
                        aux_ret = torch.stack(aux_ret)
                    elif isinstance(aux_video_processor, BaseImageProcessor):
                        aux_ret = aux_video_processor([np.asarray(frame.convert(
                            'RGB')) for frame in frames], return_tensors="pt").data["pixel_values"]
                    else:
                        raise NotImplementedError("invalid aux_video_processor") 
                    ret_dict[aux_frame_key] = aux_ret
            return ret_dict
        return {self.frames_key: frames}

    def collate_frames(self, batch_data, collate_data):
        frames_list = []
        frame_len_list = []
        aux_frames_dict_of_list = {}

        for data in batch_data:
            data.pop("n_frames")
            frame_dict = {}
            if self.frames_key in data:
                frames = data.pop(self.frames_key)
                frame_len_list.append(len(frames))
                frames_list.append(frames)
            else:
                frame_len_list.append(0)

            if self.enable_aux_frames:
                for aux_frame_key in self.aux_frame_keys:
                    if aux_frame_key in data:
                        if aux_frame_key not in aux_frames_dict_of_list:
                            aux_frames_dict_of_list[aux_frame_key] = []
                        aux_frames_dict_of_list[aux_frame_key].append(frame_dict.pop(aux_frame_key))

        if len(frames_list) > 0:
            collate_data["frames"] = torch.cat(frames_list, dim=0)
        else:
            collate_data["frames"] = frames_list
        if self.enable_aux_frames:  
            for aux_frame_key in self.aux_frame_keys:
                if aux_frame_key in aux_frames_dict_of_list:
                    collate_data[aux_frame_key] = torch.cat(aux_frames_dict_of_list[aux_frame_key], dim=0)

        collate_data["n_frames"] = frame_len_list

        return batch_data, collate_data

    def batch_transform(self, batch_data):
        collate_data = {}
        try:
            batch_data, collate_data = self.collate_frames(
                batch_data, collate_data)
        except Exception as e:
            logger.error(f"Error in batch_transform: {e}")
            return None

        # collate all meta keys as list(str)
        all_keys = list(batch_data[0].keys())
        for key in all_keys:
            if isinstance(batch_data[0][key], str) or key in self.meta_keys:
                collate_data[key] = [data.pop(key) for data in batch_data]
        input_ids = [data.pop("input_ids") for data in batch_data]
        input_ids = self.padding_sequence(input_ids, value=self.pad_id)

        attention_mask = [data.pop("attention_mask") for data in batch_data]
        attention_mask = self.padding_sequence(attention_mask, value=0)
        collate_data.update(
            dict(input_ids=input_ids, attention_mask=attention_mask))

        if "labels" in batch_data[0].keys():
            label = [data.pop("labels") for data in batch_data]
            label = self.padding_sequence(label, value=self.ignore_index)
            collate_data["labels"] = label

        collate_data.update(default_collate(batch_data))
        collate_data['meta_keys'] = self.meta_keys
        collate_data['vision_placeholder_index'] = self.vision_placeholder_index
        collate_data['vision_placeholder'] = self.vision_placeholder
        return collate_data
