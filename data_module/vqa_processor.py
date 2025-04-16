import copy

from utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)


class VQAProcessor(object):
    """ VQA text processor, support format: 

        [{'from': 'human', 'value': '<image>\nWhat is the girl eating in the image?'}
         {'from': 'gpt', 'value': 'The girl in the image is eating a dessert, which appears to be a graham cracker treat or a cookie sandwich.'}
         {'from': 'human', 'value': "Describe the girl's hair color and clothing."}
         {'from': 'gpt', 'value': 'The girl has blonde hair, and she is wearing a pink shirt.'}]
    """
 
    def __init__(self,
                 key,
                 vision_placeholder='',
                 system_message=None,
                 system_start="<|im_start|>system\n",
                 system_end="<|im_end|>",
                 roles=("\n<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
                 ):
        self.key = key
        self.roles = roles
        self.vision_placeholder = vision_placeholder
        self.system_message = system_message or ""
        self.system_start = system_start
        self.system_end = system_end

    def add_vision_placeholders_in_prompt(self, question, data_dict):
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

        image_mode_prompt = _add_timestamp(data_dict['n_frames'])
        vision_token_exist = False

        if DEFAULT_VIDEO_TOKEN in question:
            vision_token_exist = True
            question = question.replace(
                DEFAULT_VIDEO_TOKEN, image_mode_prompt)
        elif DEFAULT_IMAGE_TOKEN in question:
            vision_token_exist = True

        if not vision_token_exist:
            # add vision token to the beginning of the prompt
            question = image_mode_prompt + question
        return question

    def __call__(self, data_dict):
        messages = copy.deepcopy(data_dict.get(self.key, []))

        system_message = self.system_message

        if messages[0]["from"] == "system":
            system_message = messages[0]["value"]
            messages = messages[1:]

        q_str_list, a_str_list = [], []

        system_message = self.system_start + system_message + self.system_end

        for i in range(0, len(messages), 2):
            question = messages[i]["value"]
            if i == 0:
                if data_dict.get('has_frame', False):
                    question = self.add_vision_placeholders_in_prompt(question, data_dict)
                question = system_message + self.roles[0] + question + self.roles[1]
            else:
                question = self.roles[0] + question + self.roles[1]

            answer = messages[i + 1]["value"]
            q_str_list.append(question)
            a_str_list.append(answer)

        return q_str_list, a_str_list
