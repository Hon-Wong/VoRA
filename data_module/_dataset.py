import os
from datasets import load_dataset, concatenate_datasets
import numpy as np

from utils import logging
from utils.constants import FILEEXT2TYPE


logger = logging.get_logger(__name__)


def _get_modality_group_indices(dataset):
    data_types = np.array(dataset["data_type"])
    global_indices = np.arange(len(data_types))
    unique_values = np.unique(data_types)

    group_indices = []
    for value in unique_values:
        mask = (data_types == value)
        group_indices.append(global_indices[mask].tolist())
    return group_indices


def get_dataset(data_paths, img_dirs, split, processor, **data_args):
    """
    从指定路径加载数据集，并对其进行预处理。

    参数:
    data_paths (list or str): 数据集文件的路径列表或单个路径。
    img_dirs (list or str): 图像目录的路径列表或单个路径。
    split (str): 要加载的数据集，例如 "train", "test" 等。
    processor: 用于处理数据集的处理器对象。
    **data_args: 其他可选的数据集参数，如 num_samples, preprocessing_batch_size 等。

    返回:
    dataset: 处理后的数据集对象。
    datasets_length (list): 每个单独加载的数据集的长度列表。
    modality_group_indices (list): 按数据类型（纯文本，图片，视频）分组的数据集索引列表。
    """

    if not isinstance(data_paths, list):
        data_paths = [data_paths]
    if not isinstance(img_dirs, list):
        img_dirs = [img_dirs]

    fileext = os.path.splitext(data_paths[0])[-1][1:]
    data_type = FILEEXT2TYPE.get(fileext, None)
    if data_type is None:
        raise ValueError(
            f"Please check your data: {fileext} is not allowed while only {FILEEXT2TYPE.keys()} are allowed.")

    all_datasets = []
    datasets_length = []
    for data_path, img_dir in zip(data_paths, img_dirs):
        dataset = load_dataset(
            path=data_type,
            data_files=data_path,
            split=split
        )
        datasets_length.append(len(dataset))
        logger.info(f"load data from: {data_path}; length: {len(dataset)}")

        def add_img_dir(example):
            example["img_dir"] = img_dir
            return example

        dataset = dataset.map(add_img_dir)
        all_datasets.append(dataset)
    dataset = concatenate_datasets(all_datasets)

    num_samples = data_args.get("num_samples", 0)
    if num_samples > 0:
        num_samples = min(len(dataset), num_samples)
        logger.info(f"Use {num_samples} samples while the length of dataset is {len(dataset)}")
        dataset = dataset.select(range(num_samples))

    column_names = list(next(iter(dataset)).keys())
    dataset = dataset.map(
        processor.batch_process,
        remove_columns=column_names,
        batched=True,
        batch_size=data_args.get("preprocessing_batch_size", 16),
        # num_proc=data_args.get("preprocessing_num_workers", 8),
        desc="Running tokenizer on dataset"
    )

    modality_group_indices = _get_modality_group_indices(dataset)
    return dataset, datasets_length, modality_group_indices
