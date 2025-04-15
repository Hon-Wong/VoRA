import ast
from pathlib import Path
import sys

import yaml


def safe_literal_eval(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "none":
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _update_nested_dict(d, keys, value):
    def _set_nested(d, keys, value):
        if len(keys) == 1:
            d[keys[0]] = value
        else:
            if keys[0] not in d:
                d[keys[0]] = {}
            _set_nested(d[keys[0]], keys[1:], value)
    _set_nested(d, keys, value)


def get_args_dict():
    args_dict = yaml.safe_load(Path(sys.argv[-1]).absolute().read_text())
    return args_dict
