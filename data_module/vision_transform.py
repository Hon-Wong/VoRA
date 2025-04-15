import math
from typing import List, Tuple, Union
import sys
from PIL import Image
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms as T
from transformers import AutoImageProcessor


class HFImageTransform:
    def __init__(self, path):
        self.image_processor = AutoImageProcessor.from_pretrained(path)

    def __call__(self, image: Image.Image):
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        return image


class PILToNdarray:
    def __init__(self):
        pass

    def __call__(self, image: Image.Image):
        image_array = np.array(image)
        return image_array.astype(np.float32)


class Rescale:
    def __init__(self, rescale_factor: float):
        self.scale = rescale_factor

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")
        return image * self.scale


class PILExpand2Square:
    def __init__(self, background_color=(0, 0, 0)):
        self.background_color = background_color

    def expand2square(self, pil_img: Image.Image):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __call__(self, image: Image.Image):
        return self.expand2square(image)


class ResizeWithAspectRatio:
    def __init__(self, size: Union[List[int], Tuple[int, int]], padding_value: float = 0, resampling=Image.BILINEAR):
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ValueError("Size must be a list or tuple with two elements: (height, width).")
        self.target_size = size
        self.padding_value = padding_value
        self.resampling = resampling

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")

        original_height, original_width = image.shape[:2]
        target_height, target_width = self.target_size

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((new_width, new_height), self.resampling))

        # Create a new image with the target size and padding value
        new_image = np.full((target_height, target_width, image.shape[2]), self.padding_value, dtype=np.float32)

        # Place the resized image in the center
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        new_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        return new_image.astype(np.float32)


def smart_resize(
    height: int, width: int, factor: int = 14, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 50 * 80 
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class PILSmartResize:
    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 1,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 50 * 80,
        resampling: int = Image.Resampling.BICUBIC,  
    ):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.resampling = resampling

    def __call__(self, image: Image.Image) -> Image.Image:
        height, width = image.height, image.width
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height), resample=self.resampling)
        return image


def create_transform(ops_cfg):
    current_module = sys.modules[__name__]
    transform_list = []
    for op in ops_cfg:
        kwargs = ops_cfg[op]
        if hasattr(T, op):
            transform_list.append(getattr(T, op)(**kwargs))
        elif hasattr(current_module, op):
            transform_list.append(getattr(current_module, op)(**kwargs))
        else:
            raise RuntimeError(f'no op {op} in torchvision.transforms and data.processors.image_transform')

    return Compose(transform_list)


class VisionProcessor:
    def __init__(self, ops):
        self.transform = create_transform(ops)

    def __call__(self, data):
        if isinstance(data, list):
            return [self.transform(_data) for _data in data]
        return self.transform(data)
