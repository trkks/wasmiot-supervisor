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


class ProgramCounterExceeded(Exception):
    '''Raised when a deployment sequence is exceeded'''

class RequestFailed(Exception):
    '''Raised when a chained request to a thing fails'''

class NotAPointer(Exception):
    '''Raised when a WebAssembly output should be a pointer and length but is
    not'''

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

    def _get_target(self, module_id, function_name):
        '''
        Return the target of this module's function's output based on
        instructions.
        '''
        return self.instructions["modules"][module_id][function_name]['to']

    def call_chain(self, output, module_id, function_name) -> CallData:
        '''
        Call a sequence of functions in order, passing the result of each to the
        next.

        Return the result of the recursive call chain or the local result which
        starts unwinding the chain.
        '''

        # Get what the given result is expected to be at next receiver.
        module = self.modules[module_id]
        response_content = list(
            module.description['paths'][f'/{{deployment}}/modules/{{module}}/{function_name}']['get']['responses']['200']['content'].items()
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
        target_url = target['servers'][0]['url'].rstrip("/") + '/' + target_path.lstrip("/")
        target_method = list(target_path_obj.keys())[0]

        return CallData(target_url, target_method, func_out_media_type, expected_result)

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
        return wu.read_from_memory(pointer, length)

    if expected_media_type == 'application/json':
        # TODO: Validate based on schema.
        try:
            block = read_bytes()
        except IndexError:
            # Interpret to JSON string as is.
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
        if expected_schema["type"] == "integer":
            return func_result
        return read_bytes()

    raise NotImplementedError(f'Unsupported response media type {expected_media_type}')