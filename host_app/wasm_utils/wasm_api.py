"""General interface for Wasm utilities."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias


ByteType: TypeAlias = bytes | bytearray

WasmType = int | float
"""Types that can be used in WebAssembly."""

class WasmRuntimeNotSetError(RuntimeError):
    """Error raised when trying to use a functionality which requires a runtime
       but the runtime has not been set properly."""


class IncompatibleWasmModule(RuntimeError):
    """Error raised when trying to mix incompatible Wasm modules."""


class WasmRuntime:
    """Superclass for Wasm runtimes."""
    def __init__(self) -> None:
        self._modules: Dict[str, WasmModule] = {}
        self._functions: Optional[Dict[str, WasmModule]] = None
        self._current_module_name: Optional[str] = None

    @property
    def modules(self) -> Dict[str, WasmModule]:
        """Get the modules loaded in the Wasm runtime."""
        return self._modules

    @property
    def functions(self) -> Dict[str, WasmModule]:
        """Get the functions loaded in the Wasm runtime and their corresponding modules."""
        if self._functions is None:
            self._functions = {}
            for _, module in self._modules.items():
                for function_name in module.functions:
                    self._functions[function_name] = module

        return self._functions

    @property
    def current_module_name(self) -> Optional[str]:
        """Get the name of the current module."""
        return self._current_module_name

    @current_module_name.setter
    def current_module_name(self, module_name: Optional[str]) -> None:
        """Set the name of the current module."""
        self._current_module_name = module_name

    def load_module(self, module: ModuleConfig) -> Optional[WasmModule]:
        """Load a module into the Wasm runtime."""
        raise NotImplementedError

    def get_or_load_module(self, module: Optional[ModuleConfig]) -> Optional[WasmModule]:
        """Get a module from the Wasm runtime, or load it if it is not found."""
        if module is None:
            return None
        wasm_module = self.modules.get(module.name)
        if wasm_module is None:
            wasm_module = self.load_module(module)
        return wasm_module

    def read_from_memory(self, address: int, length: int, module_name: Optional[str] = None
    ) -> Tuple[bytes, Optional[str]]:
        """Read from the runtime memory and return the result.

        :return Tuple where the first item is the bytes inside in the requested
        block of WebAssembly runtime's memory and the second item is None if the
        read was successful and an error if not.
        """
        raise NotImplementedError

    def write_to_memory(self, address: int, bytes_data: ByteType, module_name: Optional[str] = None
    ) -> Optional[str]:
        """Write to the runtime memory.
        Return None on success or an error message on failure."""
        raise NotImplementedError

    def run_function(self, function_name: str, params: List[Any],
                     module_name: Optional[str] = None) -> Optional[Any]:
        """Runs a function in the Wasm runtime.
        If the function is not found, return None. Otherwise, returns the function result."""
        if module_name is not None:
            module = self.modules.get(module_name)
            if module is None:
                print(f"Module '{module_name}' not found!")
                return None
            return module.run_function(function_name, params)

        for _, module in self.modules.items():
            func = module._get_function(function_name)  # pylint: disable=protected-access
            if func is not None:
                # print(f"Found function {function_name} in module {module.name}")
                return module.run_function(function_name, params)
        return None


class WasmModule:
    """Superclass for Wasm modules."""
    FunctionType = Callable[..., Any]

    def __init__(self, config: ModuleConfig, runtime: WasmRuntime) -> None:
        self._id = config.id
        self._name = config.name
        self._path = config.path
        self._runtime: WasmRuntime = runtime
        self._functions: Optional[List[str]] = None

    @property
    def id(self) -> str:  # pylint: disable=invalid-name
        """Get the id of the Wasm module."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of the Wasm module."""
        return self._name

    @property
    def path(self) -> str:
        """Get the path of the Wasm module."""
        return self._path

    @property
    def runtime(self) -> Optional[WasmRuntime]:
        """Get the runtime of the Wasm module."""
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: WasmRuntime) -> None:
        """Set the runtime of the Wasm module."""
        self._runtime = runtime

    @property
    def functions(self) -> List[str]:
        """Get the names of the known functions of the Wasm module."""
        if self._functions is None:
            if self.runtime is None:
                self._functions = []
            else:
                self._functions = self._get_all_functions()
        return self._functions

    def _get_function(self, function_name: str) -> Optional[FunctionType]:
        """Get a function from the Wasm module. If the function is not found, return None."""
        raise NotImplementedError

    def _get_all_functions(self) -> List[str]:
        """Get the names of the all known functions in the Wasm module."""
        raise NotImplementedError

    def get_arg_types(self, function_name: str) -> List[type]:
        """Get the argument types of a function from the Wasm module."""
        raise NotImplementedError

    def run_function(self, function_name: str, params: List[Any]) -> Any:
        """Run a function from the Wasm module and return the result."""
        raise NotImplementedError

    def run_data_function(self, function_name: str, data_ptr_function_name: str,
                          data: ByteType, params: List[Any]) -> ByteType:
        """Run a function from the Wasm module with data and return the transformed data."""
        func = self._get_function(function_name)
        if func is None:
            print(f"Function '{function_name}' not found!")
            return bytes()
        data_ptr_func = self._get_function(data_ptr_function_name)
        if data_ptr_func is None:
            print(f"Function '{data_ptr_function_name}' not found!")
            return bytes()
        data_ptr = self.run_function(data_ptr_function_name, [])
        if not isinstance(data_ptr, int):
            print("Could not get data pointer!")
            return bytes()

        if self.runtime is None:
            raise WasmRuntimeNotSetError

        self.runtime.write_to_memory(data_ptr, data, self.name)
        _ = self.run_function(function_name, params)

        new_data, error = self.runtime.read_from_memory(data_ptr, len(data), self.name)
        if error is not None:
            print(f"Error when reading memory: {error}")
            return bytes()
        return new_data

    def upload_data(self, data: bytes, alloc_function: str) -> Tuple[int | None, int | None]:
        """Upload data to the Wasm module.
        Return (memory pointer, size) pair of the data on success, None used on failure."""
        if self.runtime is None:
            print("Runtime not set!")
            return None, None

        try:
            data_size = len(data)
            data_pointer = self.run_function(alloc_function, [data_size])
            self.runtime.write_to_memory(data_pointer, data, self.name)
            return (data_pointer, data_size)

        except RuntimeError as error:
            print("Error when trying to upload data to Wasm module!")
            print(error)
            return None, None

    def upload_data_file(self, data_file: str | Path,
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

    def run_ml_inference(self, ml_model: MLModel, data: ByteType) -> Any:
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

            result = self.run_function(
                function_name=ml_model.infer_function_name,
                params=[model_pointer, model_size, data_pointer, data_size]
            )
            print(f"Inference result: {result}")
            return result

        except RuntimeError as error:
            print(error)
            return None

    def _load_module(self) -> None:
        """Load the Wasm module into the Wasm runtime."""
        raise NotImplementedError

    def _link_remote_functions(self) -> None:
        """Link the remote functions to the Wasm module."""
        raise NotImplementedError

ModuleDescription = dict[str, Any]
"""
Describes how a module exposes its functions as remote services. Based on
OpenAPI.
"""

@dataclass
class ModuleConfig:
    """
    Dataclass for module name, file location and associated files referred to as
    'mounts'. This is what a module instance for running functions is created
    based on.
    """
    id: str  # pylint: disable=invalid-name
    name: str
    path: str
    data_files: dict[str, Path]
    ml_model: Optional[MLModel] = None
    data_ptr_function_name: str = "get_img_ptr"

    def set_model_from_data_files(self, key: str = "model.pb") -> None:
        """Sets the model using the indicated data file."""
        if key in self.data_files:
            self.ml_model = MLModel(str(self.data_files[key]))



@dataclass
class MLModel:
    """Dataclass for ML models."""
    path: str
    alloc_function_name: str = "alloc"
    infer_function_name: str = "infer_from_ptrs"
