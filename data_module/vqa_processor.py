import copy


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

    def __call__(self, data_dict):
        messages = copy.deepcopy(data_dict.get(self.key, []))

        system_message = self.system_message

        if messages[0]["from"] == "system":
            system_message = messages[0]["value"]
            messages = messages[1:]

        q_str_list, a_str_list = [], []

        system_message = self.system_start + system_message + self.system_end

        for i in range(0, len(messages), 2):
            question = self.roles[0] + messages[i]["value"] + self.roles[1]

            if i == 0:
                question = system_message + question

            answer = messages[i + 1]["value"]
            q_str_list.append(question)
            a_str_list.append(answer)

        return q_str_list, a_str_list
