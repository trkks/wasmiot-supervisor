import wasm3
import threading

from utils.configuration import remote_functions, modules
from . import wasm3_api as w3
from .wasm3_api import rt, env

class WasmModule:
    """Class for describing WebAssembly modules"""

    def __init__(self, name="", path="", size=0, paramPath="", data_ptr=""):
        self.name = name
        self.path = path
        self.size = size
        self.paramPath = paramPath
        self.data_ptr = data_ptr
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

wasm_modules = {}
for name, details in modules.items():
    wasm_modules[name] = WasmModule(name=name,
                                    path=details["path"],
                                    size=details["size"],
                                    paramPath=details["paramPath"],
                                    data_ptr=details["data_ptr"] if "data_ptr" in details else "")
#wasm_modules = {
#    #"app1": WasmModule(
#    #    "app1.wasm", 
#    #    "modules/app1.wasm",
#    #    0,
#    #    "modules/app1.json"
#    #    ),
#    "app2": WasmModule(
#        "app2.wasm",
#        "modules/app2.wasm",
#        0,
#        "modules/app2.json"
#        ),
#    "fibo": WasmModule(
#        "fibo.wasm",
#        "../modules/fibo.wasm",
#        0,
#        "modules/fibo.json",
#        ),
#    "test": WasmModule(
#        "test.wasm",
#        "../modules/test.wasm",
#        0,
#        "modules/test.json",
#        "get_img_ptr"
#        ),
#    "camera": WasmModule(
#        "camera.wasm",
#        "../modules/camera.wasm",
#        0,
#        "modules/camera.json",
#    )
#    }

def load_module(module):
    with open(module.path, "rb") as f:
        mod = env.parse_module(f.read())
        rt.load(mod)
        w3.link_functions(mod)

def run_function(fname, params):    # parameters as list: [1,2,3]
    func = rt.find_function(fname)
    if not params: return func()
    return func(*params)

def run_data_function(fname, data_ptr, data):
    func = rt.find_function(fname)
    ptr = rt.find_function(data_ptr)()
    mem = rt.get_memory(0)
    mem[ptr:ptr+len(data)] = data
    func()
    return mem[ptr:ptr+len(data)]

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
 