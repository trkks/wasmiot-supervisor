from pathlib import Path
import json
import os

CONFIG_DIR = Path('../configs').absolute()

def get_remote_functions():
    with _check_open(CONFIG_DIR / 'remote_functions.json', {}) as f:
        return json.load(f)

def get_modules():
    with _check_open(CONFIG_DIR / 'modules.json', {}) as f:
        return json.load(f)

def get_device_description():
    """
    Load description from JSON. NOTE: Fails by design if description is
    missing because no reason to continue.
    """
    with (CONFIG_DIR / 'device-description.json').open("r") as f:
        return json.load(f)

def _check_open(path, obj):
    """
    Check if path to and file at the end of it exist and if not, create them and
    write file with default contents.

    :param path: File path read if exists or otherwise write into.
    :param obj: The default object to serialize into JSON and write.
    :return: File for reading.
    """
    if not path.exists():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with path.open("x") as f:
            json.dump(obj, f)
    return path.open("r")

remote_functions = get_remote_functions()
print(remote_functions)

modules = get_modules()