import os
import threading

from utils.configuration import remote_functions, modules
from . import wasm3_api as w3
from .wasm3_api import rt, env

class WasmModule:
    """Class for describing WebAssembly modules"""

    def __init__(self, name="", path="", size=0, paramPath="", data_ptr="", model_path="", description=""):
        self.name = name
        self.path = path
        self.size = size
        self.paramPath = paramPath
        self.data_ptr = data_ptr
        self.task_handle = None
        self.model_path = model_path
        self.description = description

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
                                    data_ptr=details["data_ptr"] if "data_ptr" in details else "",
                                    model_path=details["model_path"] if "model_path" in details else ""
                                    )
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

def run_ml_model(mod_name, image_fh):
    alloc = rt.find_function("alloc")

    model = open(wasm_modules[mod_name].model_path, 'rb')
    model_size = os.path.getsize(wasm_modules[mod_name].model_path)
    model_ptr = alloc(model_size)

    image = image_fh.read()
    image_size = len(image)
    image_ptr = alloc(image_size)

    mem = rt.get_memory(0)
    mem[model_ptr:model_ptr+model_size] = model.read()
    mem[image_ptr:image_ptr+image_size] = image

    infer = rt.find_function("infer_from_ptrs")
    res = infer(model_ptr, model_size, image_ptr, image_size)
    print("Inference result:", res)
    return res

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
 
def write_to_memory(address, bytes_data):
    """
    Put bytes_data to WebAssembly runtime's memory starting from address.

    :return None if successfully written or otherwise a string describing the
    error.
    """
    try:
        wasm_memory = rt.get_memory(0)
        wasm_memory[address:address + len(bytes_data)] = bytes_data
        return None
    except Exception as err:
        return f"Could not insert input data (length {len(bytes_data)}) into to WebAssembly memory at address ({address}): {err}"

def read_from_memory(address, length_bytes, to_list=False):
    """
    Read length_bytes amount of bytes from WebAssembly runtime's memory starting
    from address

    :return Tuple where the first item is the bytes inside in the requested
    block of WebAssembly runtime's memory and the second item is None if the
    read was successful and an error if not. 
    """
    try:
        wasm_memory = rt.get_memory(0)
        block = wasm_memory[address:address + length_bytes]
        return block.tolist() if to_list else block.tobytes(), None
    except Exception as err:
        return (
            [],
            f"Reading WebAssembly memory from address {address} with length {length_bytes} failed: {err}"
        )