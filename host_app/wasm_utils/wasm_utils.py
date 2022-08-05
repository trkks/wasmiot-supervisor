import wasm3

from . import wasm3_api as w3
from .wasm3_api import rt, env

class WasmModule:
    """Class for describing WebAssembly modules"""

    def __init__(self, name="", path="", size=0, paramPath=""):
        self.name = name
        self.path = path
        self.size = size
        self.paramPath = paramPath

wasm_modules = {
    "app1": WasmModule(
        "app1.wasm", 
        "modules/app1.wasm",
        0,
        "modules/app1.json"
        ),
    "app": WasmModule(
        "app.wasm",
        "modules/app.wasm",
        0,
        "modules/app.json"
        )
    }

def start_modules():
    for name, module in wasm_modules.items():
        print('Running module: ' + name)
        with open(module.path, "rb") as f:
            mod = env.parse_module(f.read())
            rt.load(mod)
            w3.link_functions(mod)
        wasm_run = rt.find_function("_start")

        res = wasm_run()


        if res > 1:
            print(f"Result: {res:.3f}")
        else:
            print("Error")
 