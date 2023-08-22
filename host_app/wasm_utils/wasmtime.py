"""Wasmtime Python bindings."""

from __future__ import annotations
from typing import Any, List, Optional, Tuple

from wasmtime import (
    Config, Engine, Func, FuncType, Instance, Linker, Memory, Module,
    Store, ValType, WasiConfig, WasmtimeError
)

from wasm_utils.general_utils import (
    python_clock_ms, python_delay, python_print_int, python_println, python_get_temperature,
    python_get_humidity, Print, TakeImage, RpcCall
)
from wasm_utils.wasm_api import WasmRuntime, WasmModule, ModuleConfig

SERIALIZED_MODULE_POSTFIX = ".SERIALIZED.wasm"


class WasmtimeRuntime(WasmRuntime):
    """Wasmtime runtime class."""
    def __init__(self) -> None:
        super().__init__()
        self._engine = Engine(Config())
        self._store = Store(self._engine)
        self._linker = Linker(self._engine)
        self._linker.define_wasi()
        self._wasi = WasiConfig()
        self._wasi.inherit_stdout()
        self._wasi.inherit_env()
        self._store.set_wasi(self._wasi)

        self._link_remote_functions()

    @property
    def engine(self) -> Engine:
        """Get the Wasmtime engine."""
        return self._engine

    @property
    def store(self) -> Store:
        """Get the Wasmtime store."""
        return self._store

    @property
    def linker(self) -> Linker:
        """Get the Wasmtime linker."""
        return self._linker

    def load_module(self, module: ModuleConfig) -> Optional[WasmtimeModule]:
        """Load a module into the Wasm runtime."""
        if module.name in self.modules:
            print(f"Module {module.name} already loaded!")
            return self.modules[module.name]

        wasm_module = WasmtimeModule(module, self)
        self._modules[module.name] = wasm_module
        return wasm_module

    def read_from_memory(self, address: int, length: int, module_name: Optional[str] = None
    ) -> Tuple[bytes | bytearray, Optional[str]]:
        """Read from the runtime memory and return the result.

        :return Tuple where the first item is the bytes inside in the requested
        block of WebAssembly runtime's memory and the second item is None if the
        read was successful and an error if not.
        """
        def read_from_module(module: WasmModule) -> Tuple[bytes | bytearray, Optional[str]]:
            module_memory = module.get_memory()
            if module_memory is None:
                raise RuntimeError(f"Module {module.name} has no memory!")
            block = module_memory.read(self.store, address, address + length)
            print(f"Read {len(block)} bytes from memory at address {address}")
            return block, None

        # TODO: check if there is a way to read from the memory without going through the modules

        error_str = ""
        if module_name is not None and module_name in self.modules:
            try:
                return read_from_module(self.modules[module_name])
            except (IndexError, RuntimeError) as error:
                print(f"Error when reading memory from module {module_name}: {error}")
                error_str = str(error)
        else:
            for _, wasm_module in self.modules.items():
                try:
                    return read_from_module(wasm_module)
                except (IndexError, RuntimeError) as error:
                    print(f"Error when reading memory from module {wasm_module.name}: {error}")
                    error_str = str(error)

        return (
            bytes(),
            (
                f"Reading WebAssembly memory from address {address} "
                f"with length {length} failed: {error_str}"
            )
        )

    def write_to_memory(self, address: int, bytes_data: bytes, module_name: Optional[str] = None
    ) -> Optional[str]:
        """Write to the runtime memory.
        Return None on success or an error message on failure."""
        def write_to_module(module: WasmModule) -> Optional[str]:
            module_memory = module.get_memory()
            module_memory.write(self.store, bytes_data, start=address)
            return None

        # TODO: check if there is a way to write to the memory without going through the modules

        error_str = ""
        if module_name is not None and module_name in self.modules:
            try:
                return write_to_module(self.modules[module_name])
            except (IndexError, RuntimeError) as error:
                print(f"Error when writing memory using module {module_name}: {error}")
                error_str = str(error)
        else:
            for _, wasm_module in self.modules.items():
                try:
                    return write_to_module(wasm_module)
                except (IndexError, RuntimeError) as error:
                    print(f"Error when writing memory using module {wasm_module.name}: {error}")
                    error_str = str(error)

        return (
            f"Could not insert data (length {len(bytes_data)}) into to " +
            f"WebAssembly memory at address ({address}): {error_str}"
        )

    def _link_remote_functions(self) -> None:
        sys = "sys"
        communication = "communication"
        dht = "dht"
        camera = "camera"

        i32: ValType = ValType.i32()
        f32: ValType = ValType.f32()

        # system functions
        self.linker.define_func(sys, "millis", FuncType([], [i32]), python_clock_ms)
        self.linker.define_func(sys, "delay", FuncType([i32], []), python_delay)
        self.linker.define_func(sys, "print", FuncType([i32, i32], []), Print(self).function)
        self.linker.define_func(sys, "println", FuncType([i32], []), python_println)
        self.linker.define_func(sys, "printInt", FuncType([i32, i32, i32, i32], []), python_print_int)

        # communication
        rpc_call = RpcCall(self).function
        self.linker.define_func(communication, "rpcCall", FuncType([i32, i32, i32, i32], []), rpc_call)

        # peripheral
        take_image = TakeImage(self).function
        self.linker.define_func(camera, "takeImage", FuncType([i32], []), take_image)
        self.linker.define_func(dht, "getTemperature", FuncType([], [f32]), python_get_temperature)
        self.linker.define_func(dht, "getHumidity", FuncType([], [f32]), python_get_humidity)


class WasmtimeModule(WasmModule):
    """Wasmtime module class."""
    def __init__(self, config: ModuleConfig, runtime: WasmtimeRuntime) -> None:
        self._module: Optional[Module] = None
        self._instance: Optional[Instance] = None
        super().__init__(config, runtime)

    def get_memory(self) -> Optional[Memory]:
        """Get the Wasmtime memory."""
        if self._instance is None:
            return None
        memory =  self._instance.exports(self.runtime.store)["memory"]
        return memory

    def _get_function(self, function_name: str) -> Optional[Func]:
        """Get a function from the Wasm module. If the function is not found, return None."""
        if self.runtime is None:
            print("Runtime not set!")
            return None

        try:
            func = self._instance.exports(self.runtime.store)[function_name]
            # print(f"Found function '{function_name}' in module '{self.name}'")
            return func
        except RuntimeError:
            print(f"Function '{function_name}' not found!")
            return None

    def _get_all_functions(self) -> List[str]:
        """Get the names of the all known functions in the Wasm module."""
        if self._module is None:
            return []
        return [
            item.name
            for item in self._module.imports
            if isinstance(item, Func) and item.name is not None
        ]

    def get_arg_types(self, function_name: str) -> List[type]:
        """Get the argument types of a function from the Wasm module."""
        func = self._get_function(function_name)
        if func is None:
            return []
        return [
            arg_types[str(x)]
            for x in func.type(self.runtime.store).params
        ]

    def run_function(self, function_name: str, params: List[Any]) -> Any:
        """Run a function from the Wasm module and return the result."""
        func = self._get_function(function_name)
        if func is None:
            print(f"Function '{function_name}' not found!")
            return None

        print(f"Running function '{function_name}' with params: {params}")
        if not params:
            return func(self.runtime.store)
        return func(self.runtime.store, *params)

    def _load_module(self) -> None:
        """Load the Wasm module into the Wasm runtime."""
        if self.runtime is None:
            print("Runtime not set!")
            return
        if self.runtime.linker is None:
            print("Linker not set!")
            return

        path_serial = self.path + SERIALIZED_MODULE_POSTFIX
        try:
            # try to load the module from the serialized version
            module = Module.deserialize_file(self.runtime.engine, path_serial)
        except WasmtimeError:
            # compile the module which can be a slow process
            module = Module.from_file(self.runtime.engine, self.path)
            # write a serialized version of the module to disk for later use
            byte_module: bytearray = Module.serialize(module)
            try:
                with open(path_serial, "wb") as serialized_module:
                    serialized_module.write(byte_module)
            except IOError as error:
                print(error)

        self._module = module
        self._instance = self.runtime.linker.instantiate(self.runtime.store, module)

    def _link_remote_functions(self) -> None:
        """Link some remote functions to the Wasmtime module."""
        # Note: with Wasmtime the functions have been linked to the runtime, not the module


arg_types = {
    str(ValType.i32()): int,
    str(ValType.i64()): int,
    str(ValType.f32()): float,
    str(ValType.f64()): float
}
