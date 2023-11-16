"""
This module defines the Endpoint class representing (and constraining) how
WebAssembly functions can be called via HTTP.

The description mostly follows the OpenAPI v3.0 specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from host_app.utils import FILE_TYPES


class SchemaType(Enum):
    """OpenAPI v.3.0 schema type"""
    INTEGER = 'integer'
    STRING = 'string'
    OBJECT = 'object'

class SchemaFormat(Enum):
    """OpenAPI v.3.0 schema format"""
    BINARY = 'binary'

@dataclass
class Schema:
    """JSON Schema"""
    type: SchemaType
    format: SchemaFormat | str | None = None
    properties: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        self.type = SchemaType(self.type)
        if self.format and isinstance(self.format, str):
            self.format = SchemaFormat(self.format)

@dataclass
class MediaTypeObject:
    """OpenAPI v.3.0 media type object"""
    media_type: str
    schema: Schema | dict[str, Any]
    encoding: dict[str, str] | None = None # propertyName/mountPath -> contentType

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        if isinstance(self.schema, dict):
            self.schema = Schema(**self.schema)

@dataclass
class EndpointRequest:
    """OpenAPI v.3.0 operation minus responses"""
    parameters: list[dict[str, str | bool]]
    request_body: MediaTypeObject | dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        if isinstance(self.request_body, dict):
            self.request_body = MediaTypeObject(**self.request_body)

EndpointResponse = MediaTypeObject

@dataclass
class Endpoint:
    '''Describing an endpoint for a RPC-call'''
    url: str
    path: str
    method: str
    request: EndpointRequest | dict[str, Any]
    response: EndpointResponse | dict[str, Any]

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        if isinstance(self.request, dict):
            self.request = EndpointRequest(**self.request)
        if isinstance(self.response, dict):
            self.response = EndpointResponse(**self.response)

    @classmethod
    def validated(cls, obj: dict[str, Any]): # -> Endpoint
        """Validating constructor for Endpoint"""
        # TODO
        return obj


    def open_response_files(self):
        """
        Return mapping of mount paths (i.e. relative to module root) to
        __opened__ files, which should be later sent forward as originated from
        this endpoint.
        """
        files = {}
        if self.response.media_type:
            if self.response.media_type == 'multipart/form-data':
                execution_files = list(
                    get_supported_file_schemas(self.response.schema, self.response.encoding)
                )
                # Map the mount names to files created during previous Wasm
                # call.
                for path, _schema in execution_files:
                    # TODO: Use schema to determine the encoding.
                    files[path] = open(path, 'rb')
            else:
                raise NotImplementedError(f"Unsupported media type: {self.response.media_type}")
            # No headers; requests will add them automatically.
        return files


def get_supported_file_schemas(schema: Schema, encoding):
    '''
    Return iterator of tuples of (path, schema) for all fields interpretable as
    files under multipart/form-data media type.
    '''

    return (
        (path, sub_schema) for path, sub_schema in
                schema.properties.items()
        if sub_schema['type'] == 'string' \
            and sub_schema['format'] == 'binary' \
            and encoding[path]['contentType'] in FILE_TYPES
    )

