import re


class WithNone():
    def __enter__(self):
        pass

    def __exit__(self, err_type, err_val, err_pos):
        pass


def clear_markup(s: str):
    """
    Remove square brackets and their contents.
    """
    return re.sub(r'(\[.*?\])', '', s)


def apply_format_to_float(val, fmt: str):
    if isinstance(val, float):
        return fmt.format(val)
    elif isinstance(val, dict):
        return {k: apply_format_to_float(v, fmt) for k, v in val.items()}
    elif isinstance(val, list):
        return [apply_format_to_float(v, fmt) for v in val]
    else:
        return val
