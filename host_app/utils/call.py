"""Utilities for calling endpoints"""

from functools import reduce
from dataclasses import dataclass
from typing import Any

import requests

from .endpoint import Endpoint
from .mount import MountPathFile


EndpointArgs = str | list[str] | dict[str, Any] | None
EndpointData = list[MountPathFile] | None
"""List of mount names that the module defines as outputs of a ran function"""

@dataclass
class CallData:
    '''Contains the data needed for calling a remote function's endpoint.'''
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


def call_endpoint(call: CallData, module_name: str):
    '''
    Make the provided call (with possible input files from given module's
    mounts) to another supervisor endpoint and return its response
    '''
    headers = call.headers

    # FIXME: Importing here to avoid circular imports.
    from host_app.flask_app.app import module_mount_path
    files = {
        # NOTE: IIUC opening the files here matches how 'requests' documentation
        # instructs to do for multi-file uploads, so I'm guessing it closes the
        # opened files once done.
        p.name: open(p, "rb")
        for p
        in map(
            lambda x: module_mount_path(module_name, x.path),
            call.files
        )
    }

    resp = getattr(requests, call.method)(
        call.url,
        timeout=10,
        files=files,
        headers=headers,
    )
    return resp

