'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass, field
from functools import reduce
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
from flask import current_app
import numpy as np

from host_app.wasm_utils.wasm_api import WasmRuntime, WasmModule, ModuleConfig, WasmType


FILE_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg"
]
"""
Media types that are considered files in chaining requests and thus will be
sent with whatever the sender (requests-library) decides.
"""

def module_mount_path(module_name: str, filename: str | None = None) -> Path:
    """
    Return path for a file that will eventually be made available for a
    module
    """
    return Path(current_app.config['PARAMS_FOLDER']) / module_name / (filename if filename else "")

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
        path_and_obj = assert_single_pop(description['paths'].items())
        if path_and_obj is None:
            # TODO: Can this happen? And what should be the defaults?
            path, path_and_obj = "", {}
        path, path_obj = path_and_obj

        target_method, operation_obj = get_operation(path_obj)
        response_media = get_main_response_content_entry(operation_obj)

        # Select specific media type based on the possible _input_ to this
        # endpoint (e.g., if this endpoint can receive a JPEG-image).
        request_media = None
        if (rbody := operation_obj.get('requestBody', None)):
            # NOTE: The first media type is selected.
            request_media = assert_single_pop(rbody['content'].items())

        server = assert_single_pop(description['servers'])
        if server is None:
            # TODO: Can this happen? What should be the default?
            server = {'url': ""}
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
                    ((str(y["name"]), args[str(y["name"])]) for y in endpoint.parameters),
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
                files = { 'data': open(data if data is not None else "", 'rb') }
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

    def prepare_for_running(
        self,
        module_name,
        function_name,
        args: dict,
        request_filepaths: Dict[str, str]
    ) -> Tuple[WasmModule, list[WasmType]]:
        '''
        Based on module's function's description, figure out what the
        function will need as input. And set up the module environment for
        reading or writing specified files.

        The result tuple will contain
            1. The instantiated module.
            2. Ordered arguments for the function.
        '''
        # Initialize the module.
        module = self.runtime.get_or_load_module(self.modules[module_name])
        if module is None:
            raise RuntimeError("Wasm module could not be loaded!")

        # Get the OpenAPI description for interpreting more complex arguments
        # (e.g., lists, structs or files).
        _, operation = get_operation(assert_single_pop(
            self.instructions['modules'][module_name][function_name]['paths'].values()
        ))

        # TODO: When the component model is to be integrated, map arguments in
        # request to the interface described in .wit.

        # Map the request args (query) into WebAssembly-typed (primitive)
        # arguments in an ordered list.
        types = module.get_arg_types(function_name)
        primitive_args = [t(arg) for arg, t in zip(args.values(), types)]

        # Check that all the expected media types are supported.
        # TODO: Could be done at deployment time.
        found_unsupported_medias = list(filter(
            lambda x: x not in FILE_TYPES,
            operation.get('requestBody', {})
                .get('content', {})
                .keys()
        ))

        if found_unsupported_medias:
            raise NotImplementedError(f'Input file types not supported: "{found_unsupported_medias}"')

        # Get a list of expected file parameters. The 'name' is actually
        # interpreted as a path relative to module root.
        param_files = [
            (str(parameter['name']), bool(parameter.get('required', False)))
            for parameter in operation.get('parameters', [])
            if parameter.get('in', None) == 'requestBody' and parameter.get('name', '') != ''
        ]
        request_body_files = (
            [
                # All 'pathed' files are required by default, and saving the
                # schema could be useful (encoding etc.)
                (path, schema | True)
                for path, schema in
                    get_file_schemas(
                        assert_single_pop(
                            operation['requestBody']
                                .get('content', {})
                        )
                    )
            ]
            if ('multipart/form-data' in
                operation.get('requestBody', {})
                    .get('content', {})
                    .keys())
            else []
        )

        # Lastly if the _response_ contains files, the matching filepaths need
        # to be made available for the module to write as well.
        response_media_type, response_media_schema = get_main_response_content_entry(operation)
        response_files = (
            [
                # All response files are required by default as well.
                (path, schema | True)
                for path, schema in
                    # HACK: Fake the schema field because the way the functions
                    # are atm.
                    get_file_schemas({ "schema": response_media_schema })
            ]
            if response_media_type == 'multipart/form-data'
            else []
        )

        if response_media_type == 'multipart/form-data' and \
            response_media_schema['contentMediaType'] in FILE_TYPES:
            response_files = []

        # Now actually handle mounting the files.
        file_params = param_files + request_body_files + response_files

        # Map all kinds of file parameters (optional or required) to actual
        # files.
        files_to_mount = set()
        for request_field, temp_path in request_filepaths.items():
            if (request_field, temp_path) not in files_to_mount:
                files_to_mount.add((request_field, temp_path))
            else:
                raise RuntimeError(f'Input file "{temp_path}" already mapped to "{request_field}"')

        # Check that required files are received.
        required_files = set(filter(lambda x: x[1], file_params))
        required_non_mounted_files = required_files - files_to_mount
        if required_non_mounted_files:
            raise RuntimeError(f'required input files not found:  {required_non_mounted_files}')

        # NOTE: Assuming the 'data files' have already at deployment time been
        # saved at required paths.
        # Set up the files given as input according to the paths specified in
        # request remapping fields to temporary paths and then to module's
        # expected paths.
        for module_path, temp_path in files_to_mount:
            with open(module_mount_path(module_name, module_path), "wb") as mountpath:
                with open(temp_path, "rb") as datapath:
                    mountpath.write(datapath.read())

        return module, primitive_args

    def interpret_call_from(
        self,
        module_name,
        function_name,
        wasm_out_args,
        wasm_output
    ) -> Tuple[EndpointOutput, CallData | None]:
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
                self.runtime.read_from_memory(ptr_ptr, 4, self.runtime.current_module_name)[0], byteorder='little'
            )
            length = int.from_bytes(
                self.runtime.read_from_memory(length_ptr, 4, self.runtime.current_module_name)[0], byteorder='little'
            )
            value = self.runtime.read_from_memory(ptr, length, self.runtime.current_module_name)
            as_bytes = bytes(value[0])
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
            if len(bytes_array) == 0:
                print('Empty image received')
                return None, None
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

    return str(target_method), path_obj[str(target_method).lower()]

def get_main_response_content_entry(operation_obj):
    '''
    Dig out the one and only one _response_ __media type__ and matching __schema
    object__ from under the OpenAPI v3.1.0 operation object's content field.

    NOTE: 200 is the only assumed response code.
    '''
    media_type_and_object = assert_single_pop(
        operation_obj['responses']['200']['content'].items()
    )
    if media_type_and_object is None:
        return "", {}
    media_type, media_type_object = media_type_and_object
    return media_type, media_type_object.get('schema', {})

def can_be_represented_as_wasm_primitive(schema) -> bool:
    '''
    Return True if the OpenAPI schema object can be represented as a WebAssembly
    primitive.
    '''
    type_ = schema.get('type', None)
    return type_ == 'integer' or type_ == 'float'

def get_file_schemas(media_type_obj):
    '''
    Return iterator of tuples of (path, schema) for all fields interpretable as
    files under multipart/form-data media type.
    '''

    return (
        (_path, schema) for _path, schema in
                media_type_obj.get('schema', {})
                .get('properties', {})
                .items()
        if schema['type'] == 'string' and schema['contentMediaType'] in FILE_TYPES
    )