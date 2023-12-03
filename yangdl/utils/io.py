import json
import os


__all__ = [
    'dict2json',
]


def dict2json(file_name: str, d: dict) -> None:
    """Write dict to json.

    Args:
        file_name: The file name to write dict.
        d: Data dict.
    """
    assert file_name[-4:] == 'json'

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            return str(obj)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(json.dumps(d, indent=2, separators=(', ', ': '), cls=MyEncoder))

