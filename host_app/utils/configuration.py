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

def get_device_platform_info():
    """
    TODO: Load device computing-capability -info from JSON or read from device
    example by for using psutil https://github.com/giampaolo/psutil.  NOTE:
    Fails by design if description is missing because no reason to continue.
    """
    #with (CONFIG_DIR / "platform-info.json") as f:
    #    return json.load(f)

    # ~Hardcoded~ values.
    from random import random, randrange
    to_bytes = lambda gb: gb * 1_000_000_000
    return {
        "memory": {
            "bytes": to_bytes(randrange(4, 64, 4)) # Try to emulate different gi_GA_bytes of RAM.
        },
        "cpuGrading": random()
    }

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