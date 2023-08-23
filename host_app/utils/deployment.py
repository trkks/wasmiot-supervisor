'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

import json
from dataclasses import dataclass
from typing import Any

import wasm_utils.wasm_utils as wu
import wasm_utils.wasm3_api as wa


@dataclass
class CallData:
    '''Stuff needed for calling next thing in request chain'''
    url: str
    type: str
    data: Any
    method: str = 'POST'

@dataclass
class Deployment:
    '''Describing a sequence of instructions to be executed in (some) order.'''
    instructions: dict[str, dict[str, dict[str, dict[str, Any]]]]
    modules: dict[str, wu.WasmModule]
    #main_module: wu.WasmModule
    '''
    TODO: This module contains the execution logic or "script" for the whole
    application composed of modules and distributed between devices.
    '''

    def _get_target(self, module_id, function_name):
        '''
        Return the target of this module's function's output based on
        instructions.
        '''
        return self.instructions["modules"][module_id][function_name]['to']

    def call_chain(self, output, module_id, function_name, method) -> CallData:
        '''
        Call a sequence of functions in order, passing the result of each to the
        next.

        Return the result of the recursive call chain or the local result which
        starts unwinding the chain.
        '''

        # Get what the given result is expected to be at next receiver.
        module = self.modules[module_id]
        response_content = list(
            module.description['paths'][f'/{{deployment}}/modules/{{module}}/{function_name}'][method.lower()]['responses']['200']['content'].items()
        )[0]
        func_out_media_type = response_content[0]
        func_out_schema = response_content[1].get('schema')

        # From the WebAssembly function's execution, parse result into the type
        # that needs to be used as argument for next call in sequence.
        expected_result = parse_func_result(
            output,
            wa.rt.get_memory(0),
            func_out_media_type,
            func_out_schema
        )

        # Select whether to forward the result to next node (deepening the call
        # chain) or return it to caller (respond).
        target = self._get_target(module_id, function_name)
        if target is None:
            return expected_result

        # TODO: Instead of indexing to the first, find based on ending in
        # deployment-, module-id and function_name? (it would be stupid
        # string-matching though...)
        target_path, target_path_obj = list(target['paths'].items())[0]

        OPEN_API_3_1_0_OPERATIONS = ["get", "put", "post", "delete", "options", "head", "patch", "trace"];
        target_method = next(
            (x for x in target_path_obj.keys() if x.lower() in OPEN_API_3_1_0_OPERATIONS)
        )

        # NOTE: Only one parameter is supported for now (WebAssembly currently
        # does not seem to support tuple outputs (easily))
        args = f'{target_path_obj[target_method.lower()]["parameters"][0]["name"]}={expected_result}'

        # Fill in parameters for next call based on OpenAPI description.
        target_url = target['servers'][0]['url'].rstrip('/') \
            + '/' \
            + target_path.lstrip('/') \
            + f'?{args}'

        return CallData(target_url, func_out_media_type, expected_result, target_method)

def parse_func_result(func_result, memory, expected_media_type, expected_schema=None):
    '''
    Based on media type (and schema if a structure like JSON), transform given
    WebAssembly function output into the expected format.

    ## Conversion
    - If the expected format is structured (e.g., JSON)
        - If the function result is a tuple of pointer and length, read the
        equivalent structure as UTF-8 string from WebAssembly memory.
        - If the function result is a WebAssembly primitive (e.g. integer or
        float) convert to JSON string.
    - If the expected format is binary (e.g. image or octet-stream), the result
    is expected to be a tuple of pointer and length
        - If the expected format is a file, the converted bytes are written to a
        temporary file and the filepath is returned.
    .
    '''
    def read_bytes():
        pointer = func_result[0]
        length = func_result[1]
        return bytes(wu.read_from_memory(pointer, length))

    if expected_media_type == 'application/json':
        try:
            # TODO: Validate structural JSON based on schema.
            block = read_bytes()
        except TypeError:
            # Assume the result is a Wasm primitive and interpret to JSON
            # string as is.
            return json.dumps(func_result)
        return block.decode('utf-8')
    if expected_media_type == 'image/jpeg':
        # Write the image to a temp file and return the path.
        temp_img_path = 'temp_image.jpg'
        block = read_bytes()
        with open(temp_img_path, 'wb') as f:
            f.write(block)
        return temp_img_path
    if expected_media_type == 'application/octet-stream':
        return read_bytes()

    raise NotImplementedError(f'Unsupported response media type {expected_media_type}')