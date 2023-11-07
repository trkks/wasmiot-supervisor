'''
This module defines the Deployment and CallData classes.
- Deployment interprets the instructions for how to link two WebAssembly functions together.
- CallData contains the data needed for then actually calling a remote function's endpoint.
'''

from dataclasses import dataclass, field
from functools import reduce
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Set

from host_app.wasm_utils.wasm_api import ModuleConfig, WasmModule, WasmRuntime, WasmType
from host_app.utils import FILE_TYPES
from host_app.utils.endpoint import EndpointRequest, EndpointResponse, Endpoint, Schema, SchemaType
from host_app.utils.mount import MountStage, MountPathFile


EndpointArgs = str | list[str] | dict[str, Any] | None
EndpointData = list[Path] | None
EndpointOutput = Tuple[EndpointArgs, EndpointData]

@dataclass
class CallData:
    '''Endpoint with matching arguments and other request data (files)'''
    url: str
    headers: dict[str, str]
    method: str
    files: dict[str, Any]

    @classmethod
    def from_endpoint(
        cls,
        endpoint: Endpoint,
        args: EndpointArgs = None,
        files: EndpointData = None
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
        files = endpoint.get_request_body_files()

        return cls(target_url, headers, endpoint.method, files)

@dataclass
class FunctionLink:
    '''Contains how functions should be mapped between modules.'''
    from_: Endpoint | dict[str, Any]
    to: Endpoint | dict[str, Any] | None #pylint: disable=invalid-name

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        if isinstance(self.from_, dict):
            self.from_ = Endpoint(**self.from_)
        if isinstance(self.to, dict):
            self.to = Endpoint(**self.to)

FunctionEndpointMap = dict[str, Endpoint]
ModuleEndpointMap = dict[str, dict[FunctionEndpointMap]]
"""
Mapping of module names to functions and their endpoints. NOTE: This means that
a deployment can not have two modules with the same name.
"""

FunctionLinkMap = dict[str, FunctionLink]
ModuleLinkMap = dict[str, dict[FunctionLinkMap]]

@dataclass
class Deployment:
    '''
    Describes how (HTTP) endpoints map to environment, parameters and execution of
    WebAssembly functions and vice versa.
    '''
    id: str # pylint: disable=invalid-name
    runtime: WasmRuntime
    _modules: list[ModuleConfig]
    endpoints: ModuleEndpointMap
    _instructions: dict[str, Any]
    modules: dict[str, ModuleConfig] = field(init=False)
    instructions: ModuleLinkMap = field(init=False)

    def __post_init__(self):
        self.modules = { m.name: m for m in self._modules }

        # Make the received whatever data into objects at runtime because of
        # dynamic typing.
        for module_id, functions in self.endpoints.items():
            for function_name, endpoint in functions.items():
                self.endpoints[module_id][function_name] = Endpoint(**endpoint)

        # NOTE: This is what current implementation sends as instructions which
        # might change to not have the 'module' key at all.
        self.instructions = {}
        for module_name, functions in self._instructions['modules'].items():
            self.instructions[module_name] = {}
            for function_name, link in functions.items():
                # NOTE: The from-keyword prevents using the double splat
                # operator for key-value initialization of this class.
                self.instructions[module_name][function_name] = \
                    FunctionLink(from_=link["from"], to=link["to"])

    def _next_target(self, module_name, function_name) -> Endpoint | None:
        '''
        Return the target where the module's function's output is to be sent next.
        '''

        # TODO: Check if the endpoint is on this device already or not to
        # prevent unnecessary network requests.
        return self.instructions[module_name][function_name].to

    def mounts_for(self, module_id, function_name) -> dict[MountStage, Set[dict[str, str]]]:
        '''
        Grouped by the mount stage, get the sets of files to be mounted for the
        module's function and whether they are mandatory or not.
        '''

        # TODO: When the component model is to be integrated, map arguments in
        # request to the interface described in .wit.

        request: EndpointRequest = self.endpoints[module_id][function_name].request
        if request.request_body and request.request_body.media_type == 'multipart/form-data':
            request_body_paths = (
                MountPathFile.list_from_multipart(
                    request.request_body,
                    stage=MountStage.EXECUTION
                )
            )
        else:
            # Only multipart/form-data is supported for file mounts.
            request_body_paths = []

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
            for parameter in request.parameters
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
        response: EndpointResponse = self.endpoints[module_id][function_name].response
        if response.media_type == 'multipart/form-data':
            response_files = (
                MountPathFile.list_from_multipart(
                    response.response_body,
                    stage=MountStage.OUTPUT
                )
            )
        else:
            # Only multipart/form-data is supported for file mounts.
            response_files = []

        mounts = param_files + request_body_paths + response_files

        # TODO: Use groupby instead of this triple-set-threat.
        execution_stage_mount_paths: Set[str] = set(
            filter(
                lambda y: y[0].stage == MountStage.EXECUTION,
                mounts
            )
        )
        deployment_stage_mount_paths: Set[str] = set(
            filter(
                lambda y: y[0].stage == MountStage.DEPLOYMENT,
                mounts
            )
        )
        output_stage_mount_paths: Set[str] = set(
            filter(
                lambda y: y[0].stage == MountStage.OUTPUT,
                mounts
            )
        )

        return {
           MountStage.EXECUTION: execution_stage_mount_paths,
           MountStage.DEPLOYMENT: deployment_stage_mount_paths,
           MountStage.OUTPUT: output_stage_mount_paths,
        }

    def _connect_request_files_to_mounts(
            self,
            module: WasmModule,
            function_name,
            request_filepaths: dict[str, Path]
    ) -> None:
        """
        Check the validity of and set up the given mounts for the module to use
        based on files found in request.
        """
        mounts = self.mounts_for(module.id, function_name)
        deployment_stage_mount_paths = mounts[MountStage.EXECUTION]
        execution_stage_mount_paths = mounts[MountStage.EXECUTION]

        # Map all kinds of file parameters (optional or required) to expected
        # mount paths and actual files _once_.
        # NOTE: Assuming the deployment filepaths have been handled already.
        received_filepaths: Set[str] = set(map(lambda x: x.mount_path, deployment_stage_mount_paths))
        for request_mount_path, temp_path in request_filepaths.items():
            # Check that the file is expected.
            if request_mount_path not in map(lambda x: x.mount_path, execution_stage_mount_paths):
                raise RuntimeError(f'Unexpected input file "{request_mount_path}"')

            # Check that the file is not already mapped. NOTE: This prevents
            # overwriting deployment stage files.
            if request_mount_path not in received_filepaths:
                received_filepaths.add(request_mount_path)
            else:
                raise RuntimeError(f'Input file "{temp_path}" already mapped to "{request_mount_path}"')

        # Get the paths of _required_ files.
        required_input_mount_paths: Set[str] = set(filter(
            lambda x: x.required,
            deployment_stage_mount_paths | execution_stage_mount_paths
        ))
        # Check that required files have been correctly received. Output paths
        # are not expected in request at all.
        required_but_not_mounted = required_input_mount_paths - received_filepaths
        if required_but_not_mounted:
            raise RuntimeError(f'required input files not found:  {required_but_not_mounted}')

        # Set up _all_ the files needed for this run, remapping expected mount
        # paths to temporary paths and then moving the contents between them.
        all_mounts = execution_stage_mount_paths | deployment_stage_mount_paths | mounts[MountStage.OUTPUT]
        for mount in all_mounts:
            # Search for the actual filepath in filesystem first from request
            # (execution) and second from module config (deployment).
            if temp_path := request_filepaths.get(
                mount.mount_path,
                module.get_mount(mount.mount_path)
            ):
                # FIXME: Importing here to avoid circular imports.
                from host_app.flask_app.app import module_mount_path
                host_path = module_mount_path(module.name, mount.mount_path)
                if host_path != temp_path:
                    with open(host_path, "wb") as mountpath:
                        with open(temp_path, "rb") as datapath:
                            mountpath.write(datapath.read())
                else:
                    print('File already at mount location:', host_path)
            else:
                print(f'Module expects mount "{mount.mount_path}", but it was not found in request or deployment.')



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
        self._connect_request_files_to_mounts(module, function_name, request_filepaths)

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
        source_endpoint_result = self.parse_endpoint_result(
            module_name, function_name, wasm_output
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
            module_name,
            function_name,
            func_result,
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

        from_endpoint: EndpointResponse = self.instructions[module_name][function_name].from_.response

        if from_endpoint.media_type == 'application/json':
            if can_be_represented_as_wasm_primitive(from_endpoint.schema):
                return json.dumps(func_result), None
            raise NotImplementedError('Non-primitive JSON from Wasm output not supported yet')
        if from_endpoint.media_type in FILE_TYPES:
            # The result is expected to be found in a file mounted to the module.
            out_img_path = self.modules[module_name].get_output_path()
            return None, out_img_path
        raise NotImplementedError(f'Unsupported response media type "{from_endpoint.media_type}"')

def can_be_represented_as_wasm_primitive(schema: Schema) -> bool:
    '''
    Return True if the OpenAPI schema object can be represented as a WebAssembly
    primitive.
    '''
    return schema.type in (SchemaType.INTEGER, )
