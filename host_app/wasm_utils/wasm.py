"""General settings and variables for Wasm."""

from typing import Dict

from wasm_utils.wasm_api import ModuleConfig
from wasm_utils.wasm3 import Wasm3Runtime as WasmRuntimeType

wasm_runtime: WasmRuntimeType = WasmRuntimeType()
wasm_modules: Dict[str, ModuleConfig] = {}
