import copy
import json

from torch.utils.data import Dataset

from utils import logging


logger = logging.get_logger(__name__)


class VoRADataset(Dataset):
    def __init__(self, data_paths, processor) -> None:
        self.processor = processor
        self.anns = []
        self._datasets_length = []
        self._modality_group_indices = []

        prefix_length = 0
        index = 0
        frame_key = processor.frames_key
        text_indices, image_indices, video_indices = [], [], []
        # TODO：改为多线程提升读取速度
        for data_path in data_paths:
            image_folder = data_path["image_folder"]
            anno_path = data_path["anno_path"]
            f = open(anno_path, "r")
            for line in f:
                try:
                    item = json.loads(line)
                except Exception as e:
                    logger.warning_rank0(e, line)
                    continue
                item["image_folder"] = image_folder
                self.anns.append(item)
                num_frames = len(item.get(frame_key, []))
                if num_frames == 0:
                    text_indices.append(index)
                elif num_frames == 1:
                    image_indices.append(index)
                else:
                    video_indices.append(index)
                index += 1
            self._datasets_length.append(len(self.anns) - prefix_length)
            prefix_length = len(self.anns)
            logger.info_rank0(f"Loading data from {data_path['anno_path']} done! The length is {self._datasets_length[-1]}")

        for indices in (text_indices, image_indices, video_indices):
            if len(indices) > 0:
                self._modality_group_indices.append(indices)

    def __len__(self):
        return len(self.anns)

    @property
    def datasets_length(self):
        return self._datasets_length

    @property
    def modality_group_indices(self):
        return self._modality_group_indices

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])
        output = self.processor.transform(item)
        if output is None:
            return self.__getitem__((idx + 1) % len(self))
        return output


def get_dataset(data_paths, processor):
    return VoRADataset(data_paths, processor)
