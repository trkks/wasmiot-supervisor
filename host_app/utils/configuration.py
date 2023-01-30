from pathlib import Path
import json
import os

# From: https://stackoverflow.com/questions/918154/relative-paths-in-python
CONFIG_DIR = Path(os.path.join(os.path.dirname(__file__), '../configs'))

def get_remote_functions():
    print(CONFIG_DIR / 'remote_functions.json')
    with (CONFIG_DIR / 'remote_functions.json').open()  as f:
        return json.load(f)

def get_modules():
    with (CONFIG_DIR / 'modules.json').open()  as f:
        return json.load(f)


remote_functions = get_remote_functions()

modules = get_modules()
