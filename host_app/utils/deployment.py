'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass, field
from functools import reduce
import json
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

from host_app.wasm_utils.wasm_api import WasmRuntime, WasmModule, ModuleConfig


FILE_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg"
]
"""
Media types that are considered files in chaining requests and thus will be
sent with whatever the sender (requests-library) decides.
"""

EndpointArgs = str | list[str] | dict[str, Any] | None
EndpointData = str | bytes | Path | None
EndpointOutput = Tuple[EndpointArgs, EndpointData]

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
    files: dict[str, Any]

    @classmethod
    def from_endpoint(
        cls,
        endpoint: Endpoint,
        args: EndpointArgs = None,
        data: EndpointData = None
    ):
        '''
        Fill in the parameters and input for an endpoint with arguments and
        data.
        '''

        # TODO: Fill in URL path.
        target_url = endpoint.url.rstrip('/')

        # Fill in URL query.
        if args:
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
                raise NotImplementedError(f'Unsupported parameter type "{type(args)}"')

            target_url += query

        headers = {}
        files = {}
        if endpoint.request_media_obj:
            if endpoint.request_media_obj[0] in FILE_TYPES:
                # If the result media type is a file, it is sent as 'data' when
                # the receiver reads 'files' from the request.
                files = { 'data': open(data, 'rb') }
                # No headers; requests will add them automatically.
            else:
                headers['Content-Type'] = endpoint.request_media_obj[0]
        return cls(target_url, headers, endpoint.method, files)


@dataclass
class Deployment:
    '''Describing a sequence of instructions to be executed in (some) order.'''
    runtime: WasmRuntime
    instructions: dict[str, dict[str, dict[str, dict[str, Any]]]]
    _modules: list[ModuleConfig]
    modules: dict[str, ModuleConfig] = field(init=False)
    #main_module: WasmModule
    '''
    TODO: This module could in the future contain the main execution logic or
    "script" for the _whole_ application composed of modules and distributed
    between devices.
    '''

    def __post_init__(self):
        self.modules = { m.name: m for m in self._modules }

    def _next_target(self, module_name, function_name) -> Endpoint | None:
        '''
        Return the target where the module's function's output is to be sent next.
        '''

        if (next_endpoint := self.instructions['modules'][module_name][function_name]['to']):
            # TODO: Check if the endpoint is on this device already or not to
            # prevent unnecessary network requests.
            # endpoint_description = self.instructions['endpoints'][next_endpoint.function_name]
            endpoint_description = next_endpoint
            return Endpoint.from_openapi(endpoint_description)

        return None

    def interpret_args_for(
        self,
        module_name,
        function_name,
        args: dict,
        input_file_paths: list
    ) -> Tuple[WasmModule, list[int], list[int]]:
        '''
        Based on module's function's description, figure out what the
        WebAssembly function will need as input (i.e., is it necessary to
        allocate memory and pass in pointers instead of the raw args).

        The result tuple will contain
            1. the instantiated module,
            2. arguments and argument pointers (if any) and
            3. output pointers (if any).
        '''
        # Initialize the module.
        module = self.runtime.get_or_load_module(self.modules[module_name])

        # Get the OpenAPI description for interpreting more complex arguments
        # (e.g., lists and files).
        _, operation = get_operation(assert_single_pop(
            self.instructions['modules'][module_name][function_name]['paths'].values()
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

        # Files attached to the module at deployment time.
        for data_file in self.modules[module_name].data_files:
            deployment_data_ptr, deployment_data_size = module.upload_data_file(data_file, 'alloc')
            if deployment_data_ptr is None or deployment_data_size is None:
                raise RuntimeError(f'Could not allocate memory for static file "{data_file}"')
            ptrs += [deployment_data_ptr, deployment_data_size]

        # Now that deployment-time parameters are set, map the more dynamic
        # parameters given in request.

        # Map the request args (query) into WebAssembly-typed (primitive)
        # arguments in an ordered list.
        types = module.get_arg_types(function_name)
        primitive_args = [t(arg) for arg, t in zip(args.values(), types)]

        # Files given as input in request.
        if 'requestBody' in operation:
            # NOTE: Assuming a single file as input.
            file_path = input_file_paths[0]
            file_ptr, file_size = module.upload_data_file(file_path, 'alloc')
            if file_ptr is None or file_size is None:
                raise RuntimeError(f'Could not allocate memory for input file "{file_path}"')
            ptrs += [file_ptr, file_size]

        # Lastly if the _response_ is not a primitive, memory needs to be
        # allocated for it as well for WebAssembly to write and host to read
        # later.
        result_ptrs = []
        def alloc_ptrs():
            '''
            Allocate WebAssembly memory for two 32bit pointers:
              - (1) for WebAssembly to _store address to_ its dynamically
              allocated memory block (i.e., this will point to a pointer)
              - (2) for WebAssembly to store the length of the dynamically
              allocated memory block
            '''

            ptr_ptr = module.run_function('alloc', [4])
            size_ptr = module.run_function('alloc', [4])
            if ptr_ptr is None or size_ptr is None:
                raise RuntimeError('Could not allocate memory for result pointers')

            return [ptr_ptr, size_ptr]

        _, response_media_schema = get_main_response_content_entry(operation)
        # Check a single primitive is expected as response or if the result
        # requires memory operations.
        if not can_be_represented_as_wasm_primitive(response_media_schema):
            # Allocate memory for the response.
            result_ptrs = alloc_ptrs()

        return module, primitive_args + ptrs, result_ptrs

    def interpret_call_from(
        self,
        module_name,
        function_name,
        wasm_out_args,
        wasm_output
    ) -> Tuple[EndpointOutput, CallData]:
        '''
        Find out the next function to be called in the deployment after the
        specified one.

        Return interpreted result of the current endpoint and instructions for
        the next call to be made if there is one.
        '''

        # Transform the raw Wasm result into the described output of _this_
        # endpoint for sending to the next endpoint.
        source_func_path = assert_single_pop(
            self.instructions['modules'][module_name][function_name]['paths'].values()
        )
        # NOTE: Assuming the actual method used was the one described in
        # deployment.
        _, operation_obj = get_operation(source_func_path)
        response_media = get_main_response_content_entry(operation_obj)

        source_endpoint_result = self.parse_endpoint_result(
            wasm_out_args, wasm_output, *response_media
        )

        # Check if there still is stuff to do.
        if (next_endpoint := self._next_target(module_name, function_name)):
            next_call = CallData.from_endpoint(
                next_endpoint, *source_endpoint_result
            )
            return source_endpoint_result, next_call

        return source_endpoint_result, None

    def parse_endpoint_result(
            self,
            func_out_args,
            func_result,
            media_type,
            schema
        ) -> EndpointOutput:
        '''
        Based on media type (and schema if a structure like JSON), transform given
        WebAssembly function output (in the form of out-parameters in arg-list and
        single primitive returned) into the expected format.

        ## Conversion
        - If the expected format is structured (e.g., JSON)
            - If the function result is a tuple of pointer and length, read the
            equivalent structure as UTF-8 string from WebAssembly memory.
            - If the function result is a WebAssembly primitive (e.g. integer or
            float) convert to JSON string.
        - If the expected format is binary (e.g. image or octet-stream), the result
        is expected to be a tuple of pointer and length
            - If the expected format is a file, the converted bytes are written to a
            temporary file and the filepath in the form of `Path` is returned.
        .
        '''
        def read_out_bytes():
            ptr_ptr = func_out_args[0]
            length_ptr = func_out_args[1]
            ptr = int.from_bytes(
                self.runtime.read_from_memory(ptr_ptr, 4), byteorder='little'
            )
            length = int.from_bytes(
                self.runtime.read_from_memory(length_ptr, 4), byteorder='little'
            )
            value = self.runtime.read_from_memory(ptr, length)
            as_bytes = bytes(value)
            return as_bytes

        if media_type == 'application/json':
            if can_be_represented_as_wasm_primitive(schema):
                return json.dumps(func_result), None

            # TODO: Validate structural JSON based on schema.
            block = read_out_bytes()
            return block.decode('utf-8'), None
        if media_type == 'image/jpeg':
            # Write the image to a temp file and return the path.
            temp_img_path = Path('temp_image.jpg')
            block = read_out_bytes()
            bytes_array = np.frombuffer(block, dtype=np.uint8)
            img = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
            cv2.imwrite(temp_img_path.as_posix(), img)
            return None, temp_img_path
        if media_type == 'application/octet-stream':
            return None, read_out_bytes()

        raise NotImplementedError(f'Unsupported response media type "{media_type}"')

def assert_single_pop(iterable) -> Any | None:
    '''
    Assert that the iterator has only one item and return it.
    '''
    iterator = iter(iterable)
    item = next(iter(iterator), None)
    assert item and next(iterator, None) is None, 'Only one item expected'
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
    Dig out the one and only one _response_ __media type__ and matching __schema
    object__ from under the OpenAPI v3.1.0 operation object's content field.

    NOTE: 200 is the only assumed response code.
    '''
    media_type, media_type_object = assert_single_pop(
        operation_obj['responses']['200']['content'].items()
    )
    return media_type, media_type_object.get('schema', {})

def can_be_represented_as_wasm_primitive(schema) -> bool:
    '''
    Return True if the OpenAPI schema object can be represented as a WebAssembly
    primitive.
    '''
    type_ = schema.get('type', None)
    return type_ == 'integer' or type_ == 'float'
