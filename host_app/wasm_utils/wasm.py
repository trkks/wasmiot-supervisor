"""General settings and variables for Wasm."""

from typing import Dict

from wasm_utils.wasm_api import ModuleConfig, MLModel
from wasm_utils.wasm3 import Wasm3Runtime as WasmRuntimeType
from wasm_utils.wasm3 import Wasm3Module as WasmModuleType

wasm_runtime: WasmRuntimeType = WasmRuntimeType()
wasm_modules: Dict[str, ModuleConfig] = {}
