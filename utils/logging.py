# This code is inspired by the hiyouga/LLaMA-Factory's logging implementation.
# https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/extras/logging.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import colorlog
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import transformers

from .constants import DEFAULT_RUNNING_LOG


_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


class LoggerHandler(logging.Handler):
    r"""Redirect the logging output to the logging file for LLaMA Board."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self._formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(filename)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.setLevel(logging.INFO)
        os.makedirs(output_dir, exist_ok=True)
        self.running_log = os.path.join(output_dir, DEFAULT_RUNNING_LOG)
        try:
            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                if os.path.exists(self.running_log):
                    os.remove(self.running_log)
        except Exception as e:
            print(f"Error removing running log: {e}")

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _write_log(self, log_entry: str) -> None:
        with open(self.running_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n\n")

    def emit(self, record) -> None:
        if record.name == "httpx":
            return

        log_entry = self._formatter.format(record)
        self.thread_pool.submit(self._write_log, log_entry)

    def close(self) -> None:
        self.thread_pool.shutdown(wait=True)
        return super().close()


def _get_default_logging_level() -> "logging._Level":
    r"""Return the default logging level."""
    env_level_str = os.environ.get("VERBOSITY", None)
    if env_level_str:
        if env_level_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_level_str.upper()]
        else:
            raise ValueError(f"Unknown logging level: {env_level_str}.")

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r"""Configure root logger using a stdout stream handler with an explicit format."""
    global _default_handler

    with _thread_lock:
        if _default_handler:  # already configured
            return

        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    r"""Return a logger with the specified name. It it not supposed to be accessed externally."""
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    # TODO: FIXME root_logger is not same
    logger = logging.getLogger(name)
    if _default_handler is not None:
        logger.addHandler(_default_handler)
    logger.addHandler(LoggerHandler("./"))
    logger.setLevel(_get_default_logging_level())
    logger.propagate = False
    return logger


def add_handler(handler: "logging.Handler") -> None:
    r"""Add a handler to the root logger."""
    _configure_library_root_logger()
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    r"""Remove a handler to the root logger."""
    _configure_library_root_logger()
    _get_library_root_logger().removeHandler(handler)


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once

# Set transformers logging level
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()