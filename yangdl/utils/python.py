import importlib
import inspect
import os
from typing import Callable


def get_properties(obj) -> dict:
    return {key: obj.__getattribute__(key) for key, val in vars(obj.__class__).items() if isinstance(val, property)}


def get_class_name_from_method(f: Callable):
    return f.__qualname__.split('.')[0]


def get_class_from_method(f: Callable):
    cls_name = get_class_name_from_method(f)
    return getattr(importlib.import_module(f.__module__), cls_name)


def method_is_overrided_in_subclass(f: Callable):
    cls = get_class_from_method(f)
    for parent in cls.__mro__[1:]:
        if f.__name__ in parent.__dict__:
            return True
    return False


def get_caller_file_name():
    frame_info = inspect.stack()[-1]
    caller_file = frame_info.filename

    return os.path.abspath(caller_file)

