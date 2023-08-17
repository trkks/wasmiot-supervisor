"""General interface for Wasm utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

ByteType: TypeAlias = bytes | bytearray


class WasmRuntime:
    """Superclass for Wasm runtimes."""

    @property
    def modules(self) -> Dict[str, WasmModule]:
        """Get the modules loaded in the Wasm runtime."""
        raise NotImplementedError

    @property
    def functions(self) -> Dict[str, WasmModule]:
        """Get the functions loaded in the Wasm runtime and their corresponding modules."""
        raise NotImplementedError

    def load_module(self, module: ModuleConfig) -> Optional[WasmModule]:
        """Load a module into the Wasm runtime."""
        raise NotImplementedError

    def get_or_load_module(self, module: ModuleConfig) -> Optional[WasmModule]:
        """Get a module from the Wasm runtime, or load it if it is not found."""
        module = self.modules.get(module.name)
        if module is None:
            module = self.load_module(module)
        return module

    def read_from_memory(self, address: int, length: int) -> ByteType:
        """Read from the runtime memory and return the result.

        :return Tuple where the first item is the bytes inside in the requested
        block of WebAssembly runtime's memory and the second item is None if the
        read was successful and an error if not.
        """
        raise NotImplementedError

    def write_to_memory(self, address: int, bytes_data: ByteType) -> Optional[str]:
        """Write to the runtime memory.
        Return None on success or an error message on failure."""
        raise NotImplementedError


class WasmModule:
    """Superclass for Wasm modules."""
    FunctionType = Callable[..., Any]

    def __init__(self, config: ModuleConfig, runtime: WasmRuntime) -> None:
        self._name = config.name
        self._path = config.path
        self._runtime: WasmRuntime = runtime
        self._functions: Optional[List[str]] = None
        self._ml_model: Optional[MLModel] = None

        self._load_module()
        self._link_remote_functions()

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

    @property
    def ml_model(self) -> Optional[MLModel]:
        """Get the ML model of the Wasm module."""
        return self._ml_model

    def get_function(self, function_name: str) -> Optional[FunctionType]:
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

    # def run_data_function(self, function_name: str, data: ByteType, params: List[Any]) -> ByteType:
    #     """Run a function from the Wasm module with data and return the transformed data."""
    #     raise NotImplementedError

    def upload_ml_model(self, ml_model: MLModel) -> Optional[Tuple[int, int]]:
        """Upload a ML model to the Wasm module.
        Return (memory pointer, size) pair of the model on success, None on failure."""
        raise NotImplementedError

    def run_ml_inference(self, data: ByteType) -> Any:
        """Run inference using the given model and data, and return the result."""
        raise NotImplementedError

    def _load_module(self) -> None:
        """Load the Wasm module into the Wasm runtime."""
        raise NotImplementedError

    def _link_remote_functions(self) -> None:
        """Link the remote functions to the Wasm module."""
        raise NotImplementedError


@dataclass
class ModuleConfig:
    """Dataclass for module name and file location."""
    name: str
    path: str
    description: Any = {}


@dataclass
class MLModel:
    """Dataclass for ML models."""
    path: str
    alloc_function_name: str = "alloc"
    infer_function_name: str = "infer_from_ptrs"
