from pathlib import Path
import json

CONFIG_DIR = Path('../configs')


def get_remote_functions():
    with (CONFIG_DIR / 'remote_functions.json').open()  as f:
        return json.load(f)

def get_modules():
    with (CONFIG_DIR / 'modules.json').open()  as f:
        return json.load(f)


remote_functions = get_remote_functions()

modules = get_modules()