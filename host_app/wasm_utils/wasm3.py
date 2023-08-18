"""Wasm3 Python bindings."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import wasm3

from wasm_utils.general_utils import (
    python_clock_ms, python_delay, python_print_int, python_println, python_get_temperature,
    python_get_humidity, Print, TakeImage, RpcCall, RandomGet
)
from wasm_utils.wasm_api import WasmRuntime, WasmModule, ModuleConfig, MLModel

RUNTIME_INIT_MEMORY = 15000


class Wasm3Runtime(WasmRuntime):
    """Wasm3 runtime class."""
    def __init__(self) -> None:
        self._env = wasm3.Environment()
        self._runtime = self._env.new_runtime(RUNTIME_INIT_MEMORY)

        self._modules: Dict[str, Wasm3Module] = {}
        self._functions: Optional[Dict[str, Wasm3Module]] = None

    @property
    def env(self) -> wasm3.Environment:
        """Get the Wasm3 environment."""
        return self._env

    @property
    def runtime(self) -> wasm3.Runtime:
        """Get the Wasm3 runtime."""
        return self._runtime

    @property
    def modules(self) -> Dict[str, Wasm3Module]:
        """Get the modules loaded in the Wasm3 runtime."""
        return self._modules

    @property
    def functions(self) -> Dict[str, Wasm3Module]:
        """Get the functions loaded in the Wasm runtime and their corresponding modules."""
        if self._functions is None:
            self._functions = {}
            for _, module in self._modules.items():
                for function_name in module.functions:
                    self._functions[function_name] = module

        return self._functions

    def load_module(self, module: ModuleConfig) -> Optional[Wasm3Module]:
        """Load a module into the Wasm runtime."""
        try:
            if module.name in self.modules:
                print(f"Module {module.name} already loaded!")
                return self.modules[module.name]

            module = Wasm3Module(module, self)
            self._modules[module.name] = module
            return module
        except AttributeError as error:
            print(error)
            return None

    def read_from_memory(self, address: int, length: int) -> Tuple[bytes, Optional[str]]:
        """Read from the runtime memory and return the result.

        :return Tuple where the first item is the bytes inside in the requested
        block of WebAssembly runtime's memory and the second item is None if the
        read was successful and an error if not.
        """
        try:
            wasm_memory = self._runtime.get_memory(0)
            block = wasm_memory[address:address + length]
            print(f"Read {len(block)} bytes from memory at address {address}")
            return block, None
        except RuntimeError as error:
            return (
                bytes(),
                (
                    f"Reading WebAssembly memory from address {address} "
                    f"with length {length} failed: {error}"
                )
            )

    def write_to_memory(self, address: int, bytes_data: bytes) -> Optional[str]:
        """Write to the runtime memory.
        Return None on success or an error message on failure."""
        try:
            wasm_memory = self.runtime.get_memory(0)
            wasm_memory[address:address + len(bytes_data)] = bytes_data
            return None
        except RuntimeError as error:
            return (
                f"Could not insert data (length {len(bytes_data)}) into to " +
                f"WebAssembly memory at address ({address}): {error}"
            )


class Wasm3Module(WasmModule):
    """Wasm3 module class."""
    def __init__(self, config: ModuleConfig, runtime: Wasm3Runtime) -> None:
        super().__init__(config, runtime)

    def get_function(self, function_name: str) -> Optional[wasm3.Function]:
        """Get a function from the Wasm module. If the function is not found, return None."""
        if self.runtime is None:
            print("Runtime not set!")
            return None

        try:
            func = self.runtime.runtime.find_function(function_name)
            print(f"Found function '{function_name}' in module '{self.name}'")
            return func
        except RuntimeError:
            print(f"Function '{function_name}' not found!")
            return None

    def _get_all_functions(self) -> List[str]:
        """Get the names of the all known functions in the Wasm module."""
        # Is there a way to get all the functions in the module with wasm3?
        if self._functions:
            return self._functions
        return []

    def get_arg_types(self, function_name: str) -> List[type]:
        """Get the argument types of a function from the Wasm module."""
        func = self.runtime.runtime.find_function(function_name)
        if func is None:
            return []
        return [
            arg_types[x]
            for x in func.arg_types
        ]

    def run_function(self, function_name: str, params: List[Any]) -> Any:
        """Run a function from the Wasm module and return the result."""
        func = self.get_function(function_name)
        if func is None:
            return None

        print(f"Running function '{function_name}' with params: {params}")
        if not params:
            return func()
        return func(*params)

    def upload_data(self, data: bytes, alloc_function: str) -> Tuple[int | None, int | None]:
        """Upload data to the Wasm module.
        Return (memory pointer, size) pair of the data on success, None used on failure."""
        if self.runtime is None:
            print("Runtime not set!")
            return None, None

        try:
            data_size = len(data)
            data_pointer = self.run_function(alloc_function, [data_size])
            self.runtime.write_to_memory(data_pointer, data)
            return (data_pointer, data_size)

        except RuntimeError as error:
            print("Error when trying to upload data to Wasm module!")
            print(error)
            return None, None

    def upload_data_file(self, data_file: str,
                         alloc_function_name: str) -> Tuple[int | None, int | None]:
        """Upload data from file to the Wasm module.
        Return (memory pointer, size) pair of the data on success, None used on failure."""
        try:
            with open(data_file, mode="rb") as file_handle:
                data = file_handle.read()
            return self.upload_data(data, alloc_function_name)

        except OSError as error:
            print("Error when trying to load data from file!")
            print(error)
            return None, None

    def upload_ml_model(self, ml_model: Optional[MLModel]) -> Tuple[int | None, int | None]:
        """Upload a ML model to the Wasm module.
        Return (memory pointer, size) pair of the model on success, None used on failure."""
        if ml_model is None:
            print("No ML model given!")
            return None, None

        return self.upload_data_file(ml_model.path, ml_model.alloc_function_name)

    def run_ml_inference(self, ml_model: MLModel, data: bytes) -> Any:
        """Run inference using the given model and data, and return the result."""
        try:
            model_pointer, model_size = self.upload_ml_model(ml_model)
            if model_pointer is None or model_size is None:
                print("Could not upload model!")
                return None
            data_pointer, data_size = self.upload_data(data, ml_model.alloc_function_name)
            if data_pointer is None or data_size is None:
                print("Could not upload data!")
                return None
            infer_function = self.get_function(ml_model.infer_function_name)
            if infer_function is None:
                print("Could not find inference function!")
                return None

            result = infer_function(model_pointer, model_size, data_pointer, data_size)
            print(f"Inference result: {result}")
            return result

        except RuntimeError as error:
            print(error)
            return None

    def _load_module(self) -> None:
        """Load the Wasm module into the Wasm runtime."""
        try:
            with open(self.path, mode="rb") as module_file:
                self._instance = self.runtime.env.parse_module(module_file.read())
            self.runtime.runtime.load(self._instance)
            self._link_remote_functions()
        except RuntimeError as error:
            print(error)

    def _link_remote_functions(self) -> None:
        """Link some remote functions to the Wasm3 module."""
        sys = "sys"
        communication = "communication"
        dht = "dht"
        camera = "camera"
        wasi = "wasi_snapshot_preview1"

        # system functions
        self._instance.link_function(sys, "millis", "i()", python_clock_ms)
        self._instance.link_function(sys, "delay", "v(i)", python_delay)
        self._instance.link_function(sys, "print", "v(*i)", Print(self.runtime).function)
        self._instance.link_function(sys, "println", "v(*)", python_println)
        self._instance.link_function(sys, "printInt", "v(i)", python_print_int)

        # communication
        rpc_call = RpcCall(self.runtime).function
        self._instance.link_function(communication, "rpcCall", "v(*iii)", rpc_call)

        # peripheral
        self._instance.link_function(camera, "takeImage", "v(i)", TakeImage(self.runtime).function)
        self._instance.link_function(dht, "getTemperature", "f()", python_get_temperature)
        self._instance.link_function(dht, "getHumidity", "f()", python_get_humidity)

        # WASI functions
        random_get = RandomGet(self.runtime).function
        self._instance.link_function(wasi, random_get.__name__, random_get)


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
