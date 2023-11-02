'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Set

from host_app.wasm_utils.wasm_api import WasmRuntime, WasmModule, ModuleConfig, WasmType


FILE_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/octet-stream"
]
"""
Media types that are considered files in chaining requests and thus will be
sent with whatever the sender (requests-library) decides.
"""

class MountStage(Enum):
    '''
    Defines the stage at which a file is mounted.
    '''
    DEPLOYMENT = 'deployment'
    EXECUTION = 'execution'

@dataclass(eq=True, frozen=True)
class MountPathFile:
    '''
    Defines the schema used for files in "multipart/form-data" requests
    '''
    mount_path: str
    media_type: str
    stage: MountStage
    encoding: str = 'base64'
    type: str = 'string'

    MediaTypeObject = dict[str, Any]

    @classmethod
    def list_from_multipart(cls, multipart: dict[str, Any]): # -> list[MountPathFile]:
        '''
        Extract list of files to mount when multipart/form-data is used to
        describe a schema of multiple files.

        Create a MountPathFiles from the JSON schema used in this project for
        describing files and their paths.
        '''
        schema = multipart['schema']
        assert schema['type'] == 'object', 'Only object schemas supported'
        assert schema['properties'], 'No properties defined for multipart schema'

        mounts = []
        for path, schema in get_file_schemas(multipart):
            media_type = multipart['encoding'][path]['contentType']
            # NOTE: The other encoding field ('format') is not regarded here.
            mount = cls(path, media_type, MountStage(schema['stage']))
            mounts.append(mount)

        return mounts

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
        Create an Endpoint object from an endpoint's (partly OpenAPI v3.1.0 path
        item object) description.
        '''
        path = description['path']
        target_method, operation_obj = validate_operation(
            description['operation']['method'], description['operation']['body']
        )
        response_media = get_main_response_content_entry(operation_obj)

        # Select specific media type based on the possible _input_ to this
        # endpoint (e.g., if this endpoint can receive a JPEG-image).
        request_media = None
        if (rbody := operation_obj.get('requestBody', None)):
            # NOTE: The first media type is selected.
            request_media = assert_single_pop(rbody['content'].items())

        url = description['url'].rstrip('/') + path

        return cls(
            url,
            target_method,
            operation_obj['parameters'],
            (response_media[0], response_media[1].get('schema', None)),
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

    def mounts_for(self, module: WasmModule, function_name: str) -> list[(MountPathFile, bool)]:
        '''
        Get the list of files to be mounted for the module's function and
        whether they are mandatory or not
        '''
        # Get the OpenAPI description for interpreting file mounts.
        operation = self.instructions['modules'][module.name][function_name]['from']['operation']['body']

        # TODO: When the component model is to be integrated, map arguments in
        # request to the interface described in .wit.

        top_level_content = operation.get('requestBody', {}).get('content', {})
        request_body_paths = (
            MountPathFile.list_from_multipart(
                top_level_content['multipart/form-data']
            )
            if 'multipart/form-data' in top_level_content
            # Only multipart/form-data is supported for file mounts.
            else []
        )

        # Check that all the expected media types are supported.
        # TODO: Could be done at deployment time.
        found_unsupported_medias = list(filter(
            lambda x: x.media_type not in FILE_TYPES, request_body_paths
        ))
        if found_unsupported_medias:
            raise NotImplementedError(f'Input file types not supported: "{found_unsupported_medias}"')

        # Get a list of expected file parameters. The 'name' is actually
        # interpreted as a path relative to module root.
        param_files = [
            (
                MountPathFile(
                    str(parameter['name']), 'application/octet-stream', MountStage.EXECUTION
                ),
                bool(parameter.get('required', False))
            )
            for parameter in operation.get('parameters', [])
            if parameter.get('in', None) == 'requestBody'
                and parameter.get('name', '') != ''
        ]

        # All 'pathed' files are required by default.
        request_body_paths = list(map(
            lambda x: (x, True),
            request_body_paths
        ))

        # Lastly if the _response_ contains files, the matching filepaths need
        # to be made available for the module to write as well.
        response_media_type, response_media_obj = get_main_response_content_entry(operation)
        response_files = (
            MountPathFile.list_from_multipart(
                response_media_obj
            )
            if 'multipart/form-data' == response_media_type
            # Only multipart/form-data is supported for file mounts.
            else []
        )

        return param_files + request_body_paths + response_files

    def prepare_for_running(
        self,
        app_context_module_mount_path,
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

        :param app_context_module_mount_path: Function for getting the path to module's mount path based on Flask app's config.
        '''
        # Initialize the module.
        module_config = self.modules[module_name]
        module = self.runtime.get_or_load_module(module_config)
        if module is None:
            raise RuntimeError("Wasm module could not be loaded!")

        # Map the request args (query) into WebAssembly-typed (primitive)
        # arguments in an ordered list.
        types = module.get_arg_types(function_name)
        primitive_args = [t(arg) for arg, t in zip(args.values(), types)]

        # Get the mounts described for this module for checking requirementes
        # and mapping to actual received files in this request.
        mounts = self.mounts_for(module, function_name)
        execution_stage_mount_paths: Set[str] = set(map(
            lambda x: x[0].mount_path,
            filter(
                lambda y: y[0].stage == MountStage.EXECUTION,
                mounts
            )
        ))
        deployment_stage_mount_paths: Set[str] = set(map(
            lambda x: x[0].mount_path,
            filter(
                lambda y: y[0].stage == MountStage.DEPLOYMENT,
                mounts
            )
        ))

        # Map all kinds of file parameters (optional or required) to expected
        # mount paths and actual files _once_.
        # NOTE: Assuming the deployment filepaths have been handled already.
        received_filepaths: Set[str] = deployment_stage_mount_paths
        for request_mount_path, temp_path in request_filepaths.items():
            # Check that the file is expected.
            if request_mount_path not in execution_stage_mount_paths:
                raise RuntimeError(f'Unexpected input file "{request_mount_path}"')

            # Check that the file is not already mapped. NOTE: This prevents
            # overwriting deployment stage files.
            if request_mount_path not in received_filepaths:
                received_filepaths.add(request_mount_path)
            else:
                raise RuntimeError(f'Input file "{temp_path}" already mapped to "{request_mount_path}"')

        # Get the paths of _required_ files.
        required_mount_paths: Set[str] = set(
            map(
                lambda y: y[0].mount_path,
                filter(lambda x: x[1], mounts)
            )
        )
        # Check that required files have been correctly received.
        required_but_not_mounted = required_mount_paths - received_filepaths
        if required_but_not_mounted:
            raise RuntimeError(f'required input files not found:  {required_but_not_mounted}')

        # Set up _all_ the files needed for this run, remapping expected mount
        # paths to temporary paths and then moving the contents between them.
        for mount, _required in mounts:
            # Search for the actual filepath in filesystem first from request
            # (execution) and second from module config (deployment).
            if temp_path := request_filepaths.get(
                mount.mount_path,
                module_config.data_files.get(mount.mount_path, None)
            ):
                host_path = app_context_module_mount_path(module_name, mount.mount_path)
                if host_path != temp_path:
                    with open(host_path, "wb") as mountpath:
                        with open(temp_path, "rb") as datapath:
                            mountpath.write(datapath.read())
                else:
                    print(f'File already at mount location:', host_path)
            else:
                print(f'Module expects mount "{mount.mount_path}", but it was not found in request or deployment.')

        return module, primitive_args

    def interpret_call_from(
        self,
        module_name,
        function_name,
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

        # NOTE: Assuming the actual method used was the one described in
        # deployment.
        operation_obj = self.instructions['modules'][module_name][function_name]['from']['operation']['body']
        response_media = get_main_response_content_entry(operation_obj)

        source_endpoint_result = self.parse_endpoint_result(
            wasm_output, response_media[0], response_media[1].get('schema', {})
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
            - If the function result is a WebAssembly primitive (e.g. integer or
            float) convert to JSON string.
        - If the expected format is binary (e.g. image or octet-stream), the
        result is expected to have been written to a file with WASI and the
        filepath in the form of `Path` is returned.
        .
        '''
        if media_type == 'application/json':
            if can_be_represented_as_wasm_primitive(schema):
                return json.dumps(func_result), None
            raise NotImplementedError('Non-primitive JSON from Wasm output not supported yet')
        if media_type == 'image/jpeg':
            temp_img_path = Path('temp_image.jpg')
            return None, temp_img_path
        raise NotImplementedError(f'Unsupported response media type "{media_type}"')

def assert_single_pop(iterable) -> Any | None:
    '''
    Assert that the iterator has only one item and return it.
    '''
    iterator = iter(iterable)
    item = next(iter(iterator), None)
    assert item and next(iterator, None) is None, 'Only one item expected'
    return item

def validate_operation(method, operation_obj) -> Tuple[str, dict[str, Any]]:
    '''
    Dig out the one and only one operation method and object from the OpenAPI
    v3.1.0 path object.
    '''
    open_api_3_1_0_operations = set((
        'get', 'put', 'post', 'delete',
        'options', 'head', 'patch', 'trace'
    ))
    assert method in open_api_3_1_0_operations, f'bad operation method: {method}'

    # TODO: Check that the operation object is valid.

    return method, operation_obj

def get_main_response_content_entry(operation_obj):
    '''
    Dig out the one and only one _response_ __media type__ and matching __media type
    object__ from under the OpenAPI v3.1.0 operation object's content field.

    NOTE: 200 is the only assumed response code.
    '''
    media_type_and_object = assert_single_pop(
        operation_obj['responses']['200']['content'].items()
    )
    if media_type_and_object is None:
        return "", {}
    media_type, media_type_object = media_type_and_object
    return media_type, media_type_object

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
        (path, schema) for path, schema in
                media_type_obj.get('schema', {})
                .get('properties', {})
                .items()
        if schema['type'] == 'string' \
            and schema['format'] == 'binary' \
            and media_type_obj['encoding'][path]['contentType'] in FILE_TYPES
    )
