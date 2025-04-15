import torch

from typing import List, Union
from PIL import Image

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .modeling_vora import VoRAForCausalLM


class VoRAProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


class VoRAProcesser(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "image_token",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_token_index = -200,
        **kwargs,
    ):
        self.image_token = image_token
        self.image_token_index = image_token_index
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[VoRAProcessorKwargs],
    ):
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        images, text = _validate_images_text_input_order(images, text)
        output_kwargs = self._merge_kwargs(
            VoRAProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            images = [[self.expand2square(image[0])] for image in images]
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}
        
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        
        input_ids = [self.tokenizer_vision_placeholder(t) for t in text]
        attention_mask = [
            [1] * len(input_ids[i]) for i in range(len(input_ids))
        ]

        text_inputs = dict(
            input_ids=torch.as_tensor(input_ids, dtype=torch.int64),
            attention_mask=torch.as_tensor(attention_mask, dtype=torch.int64),
        )
        image_inputs['frames'] = image_inputs.pop('pixel_values')
        image_inputs['n_frames'] = [len(_images) for _images in images]
        image_inputs['vision_placeholder_index'] = self.image_token_index
        return BatchFeature(data={**text_inputs, **image_inputs})

    def expand2square(self, pil_img: Image.Image):
        background_color = (0, 0, 0)
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def tokenizer_vision_placeholder(self, prompt, add_bos=False):
        def join_lists(*lists, sep):
            result = []
            for i, lst in enumerate(lists):
                if i > 0 and sep:
                    result.extend([sep])
                result.extend(lst)
            return result

        prompt_chunks = [self.tokenizer.encode(
            chunk) for chunk in prompt.split(self.image_token)]
        input_ids = join_lists(*prompt_chunks, sep=self.image_token_index)
        if add_bos:
            input_ids = [self.tokenizer.bos_token_id] + input_ids

        return input_ids


if __name__ == '__main__':
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_name = "/mnt/bn/wh-data/open_source/models/VoRA-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    conversation = [
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "url": "/mnt/bn/wh-data/data/datasets/a_demo/frames/35.jpg"
                },
                {
                    "type":"text",
                    "text":"<image> Describe this image."
                }
            ]
        }
    ]
    model_inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True).to(model.device)

    gen_kwargs = {"max_new_tokens": 1024, "pad_token_id": processor.tokenizer.eos_token_id}
    
    with torch.inference_mode():
        outputs = model.generate(model_inputs, **gen_kwargs)
        output_text = processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        print(output_text)