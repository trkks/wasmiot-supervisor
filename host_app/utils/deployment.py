'''
This module defines the Deployment class.
- Deployment interprets the instructions for how to link two WebAssembly functions together.
'''

import io
import os
from dataclasses import dataclass, field
from itertools import chain
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Set
import struct
import time

import requests
from wasmtime import WasmtimeError

from host_app.wasm_utils.wasm_api import ModuleConfig, WasmModule, WasmRuntime, WasmType
from host_app.utils import FILE_TYPES
from host_app.utils.call import CallData, EndpointArgs, EndpointData
from host_app.utils.endpoint import EndpointResponse, Endpoint, Schema, SchemaType
from host_app.utils.mount import MountStage, MountPathFile


FLASK_APP = os.environ["FLASK_APP"]
logger = logging.getLogger(FLASK_APP)

EndpointOutput = Tuple[EndpointArgs, EndpointData]

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

@dataclass
class Instructions:
    '''
    Instructions for deployment execution i.e. running and chaining functions
    together. 
    '''
    main: str | None = None
    start: str | None = None
    modules: dict[str, dict[FunctionLinkMap]] = field(default_factory=dict)

MountPathMap = dict[str, MountPathFile]
MountStageMap = dict[MountStage, list[MountPathMap]]
FunctionMountMap = dict[str, MountStageMap]
ModuleMountMap = dict[str, FunctionMountMap]

# TODO: This should probably not be a dataclass as many of its fields need
# processing post-constructor.
@dataclass
class Deployment:
    '''
    Describes how (HTTP) endpoints map to environment, parameters and execution of
    WebAssembly functions and vice versa.
    '''
    orchestrator_address: str
    id: str # pylint: disable=invalid-name
    runtimes: dict[str, WasmRuntime]
    endpoints: ModuleEndpointMap
    peers: ModuleEndpointMap
    '''Mapping to lists of peer-device execution URLs'''
    _modules: list[ModuleConfig]
    _instructions: dict[str, Any]
    _mounts: dict[str, Any]
    modules: dict[str, ModuleConfig] = field(init=False)
    instructions: Instructions = field(init=False)
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
                functions[function_name] = Endpoint(**endpoint)
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
        # Peers:
        for module_name, functions in self.peers.items():
            for function_name, endpoints in functions.items():
                endpoints_ = []
                for endpoint in endpoints:
                    endpoints_.append(Endpoint(**endpoint))
                functions[function_name] = endpoints_

        # Build how function calls are chained or linked to each other across
        # endpoints and other devices.
        self.instructions = Instructions()

        # Handle case where there is a main script.
        if "main" in self._instructions:
            self.instructions.main = self._instructions["main"]
            self.instructions.start = self._instructions["start"]

        for module_name, functions in self._instructions['modules'].items():
            self.instructions.modules[module_name] = {}
            for function_name, link in functions.items():
                self.instructions.modules[module_name][function_name] = \
                    FunctionLink(from_=link["from"], to=link["to"])

        # Add M2M functionality based on this deployment's nodes for supporting RPC inside modules.
        for mname in self.modules.keys():
            try:
                self.runtimes[mname].link_rpc_m2m(self.m2m)
            except WasmtimeError as wer:
                logger.info("Ignoring error in linking M2M to module %r:  %r", mname, wer)

        # Initialize all modules so they are ready and serialized for execution.
        for module_name, module_config in self.modules.items():
            module = self.runtimes[module_name].get_or_load_module(module_config)
            if module is None:
                raise RuntimeError("Wasm module could not be loaded!")

    @staticmethod
    def _fetch_wait_result(result_url):
        primitive_content, file_urls = None, None
        # Try a couple times, waiting for the result to be ready
        for i in range(1, 5):
            sub_result = requests.get(result_url, timeout=10)
            try:
                json_ = sub_result.json()
                primitive_content, file_urls = json_["result"]
                break
            except ValueError:
                n = i * 2
                logger.info("Sleeping %d seconds because M2M request #%d failed", n, i)
                time.sleep(n)
                pass
        else:
            logger.error("Failed result queries (last: %r)", sub_result)
            raise RuntimeError("M2M failed")
        return primitive_content, file_urls
 

    def _fetch_wait_redeployment(self, module_name, function_name, input_data):
        # Try a couple times, waiting for the result to be ready
        primitive_content, file_content = None, None
        for i in range(1, 5):
            first_resp = self._remote_procedure_call(module_name, function_name, input_data)
            # Wait in case the execution aborted because of potential redeployment.
            if first_resp.status_code != 200:
                n = i * 2
                logger.info("Sleeping %d seconds because M2M execution #%d failed", n, i)
                time.sleep(n)
                continue

            resp = first_resp.json()
            if isinstance(resp, (list, tuple)):
                # The result was returned immediately.
                primitive_content, file_content = resp
            elif isinstance(resp, dict):
                # Otherwise the result needs to be queried (when it's finished).
                result_url = resp["resultUrl"]
                sub_result = None
                primitive_content, file_urls = Deployment._fetch_wait_result(result_url)
               # TODO: Fetch and concatenate all the output files.
                file_content = []
                for url in file_urls or []:
                    file_content += requests.get(url, timeout=10).content
        return primitive_content, file_content


    def m2m(
        self,
        module_name,
        function_name,
        input_data: bytes,
        args: list[int]=[],
    ) -> Tuple[int, bytes]:
        '''
        Use peer-information in this deployment to make a request to another
        device and interpret its file-result as bytes.

        Any possible 32-bit integer WebAssembly function arguments are prepended to input_data.
        '''
        arg_bytes = bytearray()
        for arg in args:
            arg_bytes += struct.pack("<I", arg)
        input_data = arg_bytes + input_data
        primitive_content, file_content = self._fetch_wait_redeployment(module_name, function_name, input_data)
        return primitive_content, bytes(file_content)


    def _next_target(self, module_name, function_name) -> Endpoint | None:
        '''
        Return the target where the module's function's output is to be sent next.
        '''

        # TODO: Check if the endpoint is on this device already or not to
        # prevent unnecessary network requests.
        return self.instructions.modules[module_name][function_name].to

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
    
    @staticmethod
    def _remote_endpoint_call(
        remote: Endpoint,
        input_data: bytes,
        is_local: bool,
    ):
        '''Call the given remote endpoint with input and return its response'''
        # Pass possible input.
        input_args, input_files = None, None
        if len(remote.request.parameters) > 0:
            input_args = {}
            int32s = (input_data[i:i+4] for i in range(0, len(input_data), 4))
            for param, arg in zip(remote.request.parameters, int32s):
                # TODO: Instead of assuming the bytes are 4-byte unsigned integers,
                # interpret the type from attached schema.
                input_args[param["name"]] = struct.unpack("<I", arg)[0]
        if remote.request.request_body is not None:
            # Find the possible input file(name) that the target endpoint expects.

            # TODO: Should separate between file stages, but endpoint 
            # does not contain that info. HARDCODED TO SECOND ELEMENT!
            props_iter = iter(remote.request.request_body.schema.properties.keys())
            next(props_iter, None) # Skip first.
            input_mount_path = next(props_iter, None)

            input_files = {input_mount_path: io.BytesIO(input_data)} \
                if input_mount_path is not None \
                else {}

        # XXX HACK Add a flag to indicate, that this request should not be
        # directed to a peer when received.
        if is_local:
            input_args["__wasmiot_rpc"] = "local"

        # Use the CallData here just to skip some manual steps.
        call = CallData.from_endpoint(remote, input_args, files=input_files)
        resp = getattr(requests, remote.method)(
            call.url,
            timeout=10,
            files=call.files,
            headers=call.headers,
        )

        return resp


    def local_endpoint(self, module_name, function_name=None) -> Endpoint | list[Endpoint] | None:
        '''
        Return a locally existing endpoint, multiple (i.e. non empty list) if
        function_name is None.
        Otherwise None.
        '''
        endpoints = self.endpoints.get(module_name, {})
        if function_name is None and bool(endpoints):
            return list(endpoints.values())
        return endpoints.get(function_name, None)


    def _remote_procedure_call(
        self,
        module_name,
        function_name,
        input_data: bytes,
    ):
        '''
        Call a module's function (where ever it is located) with input data and
        return the response.
        '''
        # Try local endpoint first, as it would be faster.
        local_endpoint = self.local_endpoint(module_name, function_name)

        if local_endpoint is None:
            # Try remote instead.
            remote_endpoint = self.peers \
                .get(module_name, {}) \
                .get(function_name, [None])[0]
            if remote_endpoint is None:
                raise RuntimeError(f"RPC '{module_name}/{function_name}'does not exist")
            target_endpoint = remote_endpoint
            is_local = False
        else:
            target_endpoint = local_endpoint
            is_local = True

        return Deployment._remote_endpoint_call(target_endpoint, input_data, is_local)


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
            return wasm_output, [output_mounts[0]]
        raise NotImplementedError(f'Unsupported response media type "{response_endpoint.media_type}"')

def can_be_represented_as_wasm_primitive(schema: Schema) -> bool:
    '''
    Return True if the OpenAPI schema object can be represented as a WebAssembly
    primitive.
    '''
    return schema.type in (SchemaType.INTEGER, )
