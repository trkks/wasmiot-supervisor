import wasm3
import threading

from . import wasm3_api as w3
from .wasm3_api import rt, env

class WasmModule:
    """Class for describing WebAssembly modules"""

    def __init__(self, name="", path="", size=0, paramPath=""):
        self.name = name
        self.path = path
        self.size = size
        self.paramPath = paramPath
        self.task_handle = None

# wasm3 maps wasm function argument types as follows:
# i32 : 1
# i64 : 2
# f32 : 3
# f64 : 4
# Here mapped to python types for parsing from http-requests
arg_types = {
    1: int,
    2: int,
    3: float,
    4: float
}

wasm_modules = {
    #"app1": WasmModule(
    #    "app1.wasm", 
    #    "modules/app1.wasm",
    #    0,
    #    "modules/app1.json"
    #    ),
    "app2": WasmModule(
        "app2.wasm",
        "modules/app2.wasm",
        0,
        "modules/app2.json"
        ),
    "fibo": WasmModule(
        "fibo.wasm",
        "../modules/fibo.wasm",
        0,
        "modules/fibo.json",
        )
    }

def load_module(module):
    with open(module.path, "rb") as f:
        mod = env.parse_module(f.read())
        rt.load(mod)
        w3.link_functions(mod)

def run_function(fname, params):    # parameters as list: [1,2,3]
    func = rt.find_function(fname)
    return func(*params)

def get_arg_types(fname):
    func = rt.find_function(fname)
    return list(map(lambda x: arg_types[x], func.arg_types))

def start_modules():
    for name, module in wasm_modules.items():
        print('Running module: ' + name)
        with open(module.path, "rb") as f:
            mod = env.parse_module(f.read())
            rt.load(mod)
            w3.link_functions(mod)
        wasm_run = rt.find_function("_start")

        #res = wasm_run()
        module.task_handle = threading.Thread(name=name, daemon=True, target=wasm_run)
        module.task_handle.start()

        #if res > 1:
        #    print(f"Result: {res:.3f}")
        #else:
        #    print("Error")
 