'''
This module defines the Deployment and CallData classes.
- Deployment interprets the instructions for how to link two WebAssembly functions together.
- CallData contains the data needed for then actually calling a remote function's endpoint.
'''

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Set

from host_app.wasm_utils.wasm_api import ModuleConfig, WasmModule, WasmRuntime, WasmType
from host_app.utils import FILE_TYPES
from host_app.utils.endpoint import EndpointResponse, Endpoint, Schema, SchemaType
from host_app.utils.mount import MountStage, MountPathFile


EndpointArgs = str | list[str] | dict[str, Any] | None
EndpointData = list[str] | None
"""List of mount names that the module defines as outputs of a ran function"""
EndpointOutput = Tuple[EndpointArgs, EndpointData]

@dataclass
class CallData:
    '''Endpoint with matching arguments and other request data (files)'''
    url: str
    headers: dict[str, str]
    method: str
    files: list[str] | None

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
        target_url = endpoint.url.rstrip('/') + endpoint.path

        # Fill in URL query.
        if args:
            if isinstance(args, str):
                # Add the single parameter to the query.

                # NOTE: Only one parameter is supported for now (WebAssembly currently
                # does not seem to support tuple outputs (easily)). Also path should
                # have been already filled and provided in the deployment phase.
                param_name = endpoint.request.parameters[0]["name"]
                param_value = args
                query = f'?{param_name}={param_value}'
            elif isinstance(args, list):
                # Build the query in order.
                query = reduce(
                    lambda acc, x: f'{acc}&{x[0]}={x[1]}',
                    zip(map(lambda y: y["name"], endpoint.request.parameters), args),
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

        return cls(target_url, headers, endpoint.method, files or {})

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

MountPathMap = dict[str, MountPathFile]
MountStageMap = dict[MountStage, list[MountPathMap]]
FunctionMountMap = dict[str, MountStageMap]
ModuleMountMap = dict[str, FunctionMountMap]

@dataclass
class Deployment:
    '''
    Describes how (HTTP) endpoints map to environment, parameters and execution of
    WebAssembly functions and vice versa.
    '''
    id: str # pylint: disable=invalid-name
    runtimes: dict[str, WasmRuntime]
    _modules: list[ModuleConfig]
    endpoints: ModuleEndpointMap
    _instructions: dict[str, Any]
    _mounts: dict[str, Any]
    modules: dict[str, ModuleConfig] = field(init=False)
    instructions: ModuleLinkMap = field(init=False)
    mounts: ModuleMountMap = field(init=False)

    def __post_init__(self):
        # Map the modules by their names for easier access.
        self.modules = { m.name: m for m in self._modules }

        # Make the received whatever data into objects at runtime because of
        # dynamic typing. NOTE/FIXME: This is mutating the collection while
        # iterating it which might be bad...
        # Endpoints:
        for module_name, functions in self.endpoints.items():
            for function_name, endpoint in functions.items():
                self.endpoints[module_name][function_name] = Endpoint(**endpoint)
        # Mounts:
        self.mounts = {}
        for module_name, functions in self._mounts.items():
            self.mounts[module_name] = {}
            for function_name, stage_mounts in functions.items():
                self.mounts[module_name][function_name] = {}
                for stage, mounts in stage_mounts.items():
                    # NOTE: There might be duplicate paths in the mounts.
                    self.mounts[module_name][function_name][MountStage(stage)] = \
                        [MountPathFile(**mount) for mount in mounts]

        # Build how function calls are chained or linked to each other across
        # endpoints and other devices.
        self.instructions = {}
        # NOTE: This is what current implementation sends as instructions which
        # might change to not have the 'modules' key at all.
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

    def _connect_request_files_to_mounts(
        self,
        module_name,
        function_name,
        request_filepaths: dict[str, Path]
    ) -> None:
        """
        Check the validity of file mounts received in request. Set _all_ mounts
        up for the module to use for this function.

        The setup is needed, because received files in requests are saved into
        some arbitrary filesystem locations, where they need to be moved from
        for the Wasm module to access.
        """
        mounts: MountStageMap = self.mounts[module_name][function_name]
        deployment_stage_mount_paths = mounts[MountStage.DEPLOYMENT]
        execution_stage_mount_paths = mounts[MountStage.EXECUTION]

        # Map all kinds of file parameters (optional or required) to expected
        # mount paths and actual files _once_.
        # NOTE: Assuming the deployment filepaths have been handled already.
        received_filepaths: Set[str] = set(map(lambda x: x.path, deployment_stage_mount_paths))
        for request_mount_path, temp_source_path in request_filepaths.items():
            # Check that the file is expected.
            if request_mount_path not in map(lambda x: x.path, execution_stage_mount_paths):
                raise RuntimeError(f'Unexpected input file "{request_mount_path}"')

            # Check that the file is not already mapped. NOTE: This prevents
            # overwriting deployment stage files.
            if request_mount_path not in received_filepaths:
                received_filepaths.add(request_mount_path)
            else:
                raise RuntimeError(f'Input file "{temp_source_path}" already mapped to "{request_mount_path}"')

        # Get the paths of _required_ files.
        required_input_mount_paths: Set[str] = set(map(
            lambda y: y.path,
            filter(
                lambda x: x.required,
                chain(deployment_stage_mount_paths, execution_stage_mount_paths)
            )
        ))

        # Check that required files have been correctly received. Output paths
        # are not expected in request at all.
        required_but_not_mounted = required_input_mount_paths - received_filepaths
        if required_but_not_mounted:
            raise RuntimeError(f'required input files not found:  {required_but_not_mounted}')

        # Set up _all_ the files needed for this run, remapping expected mount
        # paths to temporary paths and then moving the contents between them.
        all_mounts = chain(execution_stage_mount_paths, deployment_stage_mount_paths, mounts[MountStage.OUTPUT])
        for mount in all_mounts:
            temp_source_path = None
            match mount.stage:
                case MountStage.DEPLOYMENT:
                    temp_source_path = self.modules[module_name].data_files.get(mount.path, None)
                case MountStage.EXECUTION:
                    temp_source_path = request_filepaths.get(mount.path, None)
                case MountStage.OUTPUT:
                    continue

            if not temp_source_path:
                print(f'Module expects mount "{mount.path}", but it was not found in request or deployment.')
                raise RuntimeError(f'Missing input file "{mount.path}"')

            # FIXME: Importing here to avoid circular imports.
            from host_app.flask_app.app import module_mount_path
            host_path = module_mount_path(module_name, mount.path)
            if host_path != temp_source_path:
                with open(host_path, "wb") as mountpath:
                    with open(temp_source_path, "rb") as datapath:
                        mountpath.write(datapath.read())
            else:
                print('File already at mount location:', host_path)

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

        :param app_context_module_mount_path: Function for getting the path to
        module's mount path based on Flask app's config.
        '''
        # Initialize the module.
        module_config = self.modules[module_name]
        module = self.runtimes[module_name].get_or_load_module(module_config)
        if module is None:
            raise RuntimeError("Wasm module could not be loaded!")

        # Map the request args (query) into WebAssembly-typed (primitive)
        # arguments in an ordered list.
        types = module.get_arg_types(function_name)
        primitive_args = [t(arg) for arg, t in zip(args.values(), types)]

        # Get the mounts described for this module for checking requirementes
        # and mapping to actual received files in this request.
        self._connect_request_files_to_mounts(module.name, function_name, request_filepaths)

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
        next_exec_args, next_exec_files = self.parse_endpoint_result(
            wasm_output,
            self.endpoints[module_name][function_name].response,
            self.mounts[module_name][function_name][MountStage.OUTPUT]
        )

        # Check if there still is stuff to do.
        if (next_endpoint := self._next_target(module_name, function_name)):
            next_call = CallData.from_endpoint(
                next_endpoint, next_exec_args, next_exec_files
            )
            return (next_exec_args, next_exec_files), next_call

        return (next_exec_args, next_exec_files), None

    def parse_endpoint_result(
            self,
            wasm_output,
            response_endpoint: EndpointResponse,
            output_mounts: dict[str, MountPathFile]
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

        if response_endpoint.media_type == 'application/json':
            if can_be_represented_as_wasm_primitive(response_endpoint.schema):
                return json.dumps(wasm_output), None
            raise NotImplementedError('Non-primitive JSON from Wasm output not supported yet')
        if response_endpoint.media_type in FILE_TYPES:
            # The result is expected to be found in a file mounted to the module.
            assert len(output_mounts) == 1, \
                f'One and only one output file expected for media type "{response_endpoint.media_type}"'
            out_img_name = output_mounts[0].path
            return None, [out_img_name]
        raise NotImplementedError(f'Unsupported response media type "{response_endpoint.media_type}"')

def can_be_represented_as_wasm_primitive(schema: Schema) -> bool:
    '''
    Return True if the OpenAPI schema object can be represented as a WebAssembly
    primitive.
    '''
    return schema.type in (SchemaType.INTEGER, )
