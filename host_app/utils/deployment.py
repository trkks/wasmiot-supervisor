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

    def run_function(self, module, function_name, method, args) -> CallData:
        '''
        TODO: This might make more sense to reside somewhere else than
        'deployment' module.

        Using the module description about parameters and results, call the
        function in module

        :returns The result of the function execution in its described
        format.
        '''
        output = module.run_function(function_name, args)

        # Get what format the given output is expected to be in.
        response_content = list(
            module.description['paths'][f'/{{deployment}}/modules/{{module}}/{function_name}'][method.lower()]['responses']['200']['content'].items()
        )[0]
        func_out_media_type = response_content[0]
        func_out_schema = response_content[1].get('schema')

        # Parse the WebAssembly function's execution result into the format.
        expected_result = parse_func_result(
            output,
            wa.rt.get_memory(0),
            func_out_media_type,
            func_out_schema
        )

        return expected_result

    def next_target(self, module_id, function_name):
        '''
        Return the target of this module's function's output based on
        instructions.
        '''
        return self.instructions["modules"][module_id][function_name]['to']

    def call_chain(self, target, params) -> CallData:
        '''
        Find out the next function to be called in the deployment after the
        specified one.

        Return instructions for the next call to be made or None if not needed.
        '''
        # NOTE: Assuming the deployment contains only one path for now.
        target_path, target_path_obj = list(target['paths'].items())[0]

        OPEN_API_3_1_0_OPERATIONS = ["get", "put", "post", "delete", "options", "head", "patch", "trace"]
        target_method = next(
            (x for x in target_path_obj.keys() if x.lower() in OPEN_API_3_1_0_OPERATIONS)
        )

        # Select specific media type if request input file requires it.
        media_type = None
        if rbody := target_path_obj[target_method.lower()] \
            .get('requestBody', None):
            media_type = rbody.content.keys()[0]

        # Fill in parameters for next call based on OpenAPI description.
        # NOTE: Only one parameter is supported for now (WebAssembly currently
        # does not seem to support tuple outputs (easily))
        args = f'{target_path_obj[target_method.lower()]["parameters"][0]["name"]}={params}'

        # Path (TODO) and query.
        target_url = target['servers'][0]['url'].rstrip('/') \
            + '/' \
            + target_path.lstrip('/') \
            + f'?{args}'

        return CallData(target_url, media_type, params, target_method)

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