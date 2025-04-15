## Special token and index
# image level
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
# video level
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
# vison placeholder
ONE_PLACEHOLDER_PER_IMAGE = 0
ONE_PLACEHOLDER_PER_VIDEO = 1

# ## tokenizer
IGNORE_INDEX = -100

## log
DEFAULT_RUNNING_LOG = './logs.txt'

## dataset
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

DATA_TYPE_TEXT = 0
DATA_TYPE_IMAGE = 1
DATA_TYPE_VIDEO = 2
