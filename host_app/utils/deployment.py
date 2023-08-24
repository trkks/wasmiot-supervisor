'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass
from functools import reduce
import json
import os
from typing import Any, Tuple

import wasm_utils.wasm_utils as wu
from wasm_utils.wasm3_api import rt


@dataclass
class Endpoint:
    '''Describing an endpoint for a RPC-call'''
    url: str
    method: str
    parameters: list[dict[str, str | bool]]
    response_media_obj: Tuple[str, dict[str, Any] | None]
    request_media_obj: Tuple[str, dict[str, Any] | None] | None = None

    @classmethod
    def from_openapi(cls, description: dict[str, Any]):
        '''
        Create an Endpoint object from an endpoint's OpenAPI v3.1.0
        description path item object.
        '''
        # NOTE: The deployment should only contain one path per function.
        path, path_obj = assert_single_pop(description['paths'].items())

        target_method, operation_obj = get_operation(path_obj)
        response_media = get_main_response_content_entry(operation_obj)

        # Select specific media type based on the possible _input_ to this
        # endpoint (e.g., if this endpoint can receive a JPEG-image).
        request_media = None
        if (rbody := operation_obj.get('requestBody', None)):
            # NOTE: The first media type is selected.
            request_media = assert_single_pop(rbody['content'].items())

        server = assert_single_pop(description['servers'])
        url = server['url'].rstrip('/') + path

        return cls(
            url,
            target_method,
            operation_obj['parameters'],
            response_media,
            request_media
        )

@dataclass
class CallData:
    '''Minimal stuff needed for calling an endpoint'''
    url: str
    headers: dict[str, str]
    method: str
    data: str | dict[str, Any]

    @classmethod
    def from_endpoint(
        cls,
        endpoint: Endpoint,
        args: Any,
        data: str | dict[str, Any] | None = None
    ):
        '''
        Fill in the parameters for an endpoint with arguments and data.
        '''

        # TODO: Fill in URL path.

        # Fill in URL query.
        if isinstance(args, str):
            # Add the single parameter to the query.

            # NOTE: Only one parameter is supported for now (WebAssembly currently
            # does not seem to support tuple outputs (easily)). Also path should
            # have been already filled and provided in the deployment phase.
            param_name = endpoint.parameters[0]["name"]
            param_value = args
            query = f'?{param_name}={param_value}'
        elif isinstance(args, list):
            # Build the query in order.
            query = reduce(
                lambda acc, x: f'{acc}&{x[0]}={x[1]}',
                zip(map(lambda y: y["name"], endpoint.parameters), args),
                '?'
            )
        elif isinstance(args, dict):
            # Build the query based on matching names.
            query = reduce(
                lambda acc, x: f'{acc}&{x[0]}={x[1]}',
                ((y["name"], args[y["name"]]) for y in endpoint.parameters),
                '?'
            )
        else:
            raise NotImplementedError(f'Unsupported argument type "{type(args)}"')

        target_url = endpoint.url.rstrip('/') + query

        headers = {}
        if endpoint.request_media_obj:
            headers['Content-Type'] = endpoint.request_media_obj[0]

        return cls(target_url, headers, endpoint.method, data)


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

    def _next_target(self, module_id, function_name) -> Endpoint | None:
        '''
        Return the target where the module's function's output is to be sent next.
        '''

        if (next_endpoint := self.instructions['modules'][module_id][function_name]['to']):
            # TODO: Check if the endpoint is on this device already or not to
            # prevent unnecessary network requests.
            # endpoint_description = self.instructions['endpoints'][next_endpoint.function_name]
            endpoint_description = next_endpoint
            return Endpoint.from_openapi(endpoint_description)

        return None

    def interpret_args_for(self, module_id, function_name, args: dict, input_file_paths: list) -> list[int]:
        '''
        Based on module's function's description, figure out what the
        WebAssembly function will need as input (i.e., is it necessary to
        allocate memory and pass in pointers instead of the raw args).
        '''
        def write_file(file_path) -> Tuple[int, int]:
            '''
            Write file contents to WebAssembly dynamic memory ASSUMING
            the function for allocating memory exists in the module.
            '''
            file_handle = open(file_path, 'rb')
            size = os.path.getsize(file_path)
            alloc = rt.find_function("alloc")
            ptr = alloc(size)
            mem = rt.get_memory(0)
            mem[ptr:ptr + size] = file_handle.read()
            return ptr, size

        _, operation = get_operation(assert_single_pop(
            self.instructions['modules'][module_id][function_name]['paths'].values()
        ))

        # "Pointers" here are tuples of address and length (amount of bytes),
        # but are stored "flat" for easily passing them to WebAssembly afterwards.
        # They will be used whenever function input is not a WebAssembly
        # primitive.
        # - TODO: Write primitive-typed lists and strings to memory and add
        # their pointers to the list first.
        # - If a model (TODO: Or any other files) are attached to the module at
        # deployement time, pass pointers of their contents to the function
        # first.
        # - Pointers to contents of external files come last in the list.
        ptrs: list[int] = []

        module = self.modules[module_id]
        if module.model_path:
           model_ptr, model_size = write_file(module.model_path)
           ptrs += [model_ptr, model_size]

        if 'requestBody' in operation:
            # NOTE: Assuming a single file as input.
            file_path = input_file_paths[0]
            file_ptr, file_size = write_file(file_path)
            ptrs += [file_ptr, file_size]

        # Map the request args (query) into WebAssembly-typed (primitive)
        # arguments in an ordered list.
        types = module.get_arg_types(function_name)
        primitive_args = [t(arg) for arg, t in zip(args.values(), types)]

        return primitive_args + ptrs

    def interpret_call_from(self, module_id, function_name, wasm_output) -> Tuple[Any, CallData]:
        '''
        Find out the next function to be called in the deployment after the
        specified one.

        Return interpreted result of the current endpoint and instructions for
        the next call to be made if there is one.
        '''

        # Transform the raw Wasm result into the described output of _this_
        # endpoint for sending to the next endpoint.
        source_func_path = assert_single_pop(
            self.instructions['modules'][module_id][function_name]['paths'].values()
        )
        # NOTE: Assuming the actual method used was the one described in
        # deployment.
        _, operation_obj = get_operation(source_func_path)
        response_media = get_main_response_content_entry(operation_obj)

        source_endpoint_result = parse_endpoint_result(
            wasm_output, None, *response_media
        )

        # Check if there still is stuff to do.
        if (next_endpoint := self._next_target(module_id, function_name)):
            return source_endpoint_result, CallData.from_endpoint(next_endpoint, source_endpoint_result)
        return source_endpoint_result, None

def parse_endpoint_result(func_result, memory, media_type, schema):
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

    if media_type == 'application/json':
        try:
            # TODO: Validate structural JSON based on schema.
            block = read_bytes()
        except TypeError:
            # Assume the result is a Wasm primitive and interpret to JSON
            # string as is.
            return json.dumps(func_result)
        return block.decode('utf-8')
    if media_type == 'image/jpeg':
        # Write the image to a temp file and return the path.
        temp_img_path = 'temp_image.jpg'
        block = read_bytes()
        with open(temp_img_path, 'wb') as f:
            f.write(block)
        return temp_img_path
    if media_type == 'application/octet-stream':
        return read_bytes()

    raise NotImplementedError(f'Unsupported response media type "{media_type}"')

def assert_single_pop(iterable) -> Any | None:
    '''
    Assert that the iterator has only one item and return it.
    '''
    iterator = iter(iterable)
    item = next(iter(iterator), None)
    assert item and not next(iterator, None), 'Only one item expected'
    return item

def get_operation(path_obj) -> Tuple[str, dict[str, Any]]:
    '''
    Dig out the one and only one operation method and object from the OpenAPI
    v3.1.0 path object.
    '''
    open_api_3_1_0_operations = [
        'get', 'put', 'post', 'delete',
        'options', 'head', 'patch', 'trace'
    ]
    target_method = assert_single_pop(
        (x for x in path_obj.keys() if x.lower() in open_api_3_1_0_operations)
    )

    return target_method, path_obj[target_method.lower()]

def get_main_response_content_entry(operation_obj):
    '''
    Dig out the one and only one _response_ media type and matching object from
    the OpenAPI v3.1.0 operation object's content field.

    NOTE: 200 is the only assumed response code.
    '''
    return assert_single_pop(operation_obj['responses']['200']['content'].items())
