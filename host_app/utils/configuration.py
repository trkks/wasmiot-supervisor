from pathlib import Path
import json
import os

INSTANCE_PATH = Path(os.environ.get('INSTANCE_PATH', Path.cwd()), 'instance').absolute()
"""
Path to the directory where the application instance is located.  This is
typically the directory where the application is installed and is set by the
`INSTANCE_PATH` environment variable. It should be writable by the application.
"""
CONFIG_DIR = Path(INSTANCE_PATH, 'configs').absolute()


def get_remote_functions():
    with _check_open(CONFIG_DIR / 'remote_functions.json', {}) as f:
        return json.load(f)


def get_modules():
    with _check_open(CONFIG_DIR / 'modules.json', {}) as f:
        return json.load(f)

def get_device_description():
    """
    Load supported interfaces from JSON and combine with platform hardware info.
    NOTE: Fails by design if interface description is missing because no reason
    to continue without it.
    """
    with (CONFIG_DIR / 'wasmiot-device-description.json').open("r") as f:
        description = json.load(f)
        description["platform"] = _get_device_platform_info()
        return description

def get_wot_td():
    raise NotImplementedError

def _get_device_platform_info():
    """
    TODO: Load device computing-capability -info from JSON or read from device
    example by for using psutil https://github.com/giampaolo/psutil.  NOTE:
    Fails by design if description is missing because no reason to continue.
    """
    #with (CONFIG_DIR / "platform-info.json") as f:
    #    return json.load(f)

    # TEMPORARY: Make up some data.
    from random import random, randrange
    to_bytes = lambda gb: gb * 1_000_000_000
    return {
        "memory": {
            "bytes": to_bytes(randrange(4, 64, 4)) # Try to emulate different gi_GA_bytes of RAM.
        },
        "cpu": {
            "humanReadableName": ["12th Gen Intel(R) Core(TM) i7-12700H", "AMD EPYC™ Embedded 7551", "Dual-core Arm Cortex-M0+"][randrange(0, 3)],
            "clockSpeed": {
                "Hz": random() * 3 * 1000_000_000
            }
        }
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