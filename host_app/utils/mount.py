"""
This module describes the MountPathFile class that connects a module's data
files and endpoint description(s) together into filepaths relative to the module.

The mounts are expected at different stages: deployment, execution, and output.
The first of these is for files that are mounted when the module is deployed and
thus not seen in the endpoint descriptions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from host_app.utils.endpoint import MediaTypeObject, SchemaType, get_supported_file_schemas


class MountStage(Enum):
    '''
    Defines the stage at which a file is mounted.
    '''
    DEPLOYMENT = 'deployment'
    EXECUTION = 'execution'
    OUTPUT = 'output'

@dataclass(eq=True, frozen=True)
class MountPathFile:
    '''
    Defines the schema used for files in "multipart/form-data" requests
    '''
    mount_path: str
    media_type: str
    stage: MountStage
    required = True
    encoding: str = 'base64'
    type: str = 'string'

    @classmethod
    def validate(cls, x: dict[str, Any]): # -> MountPathFile:
        # TODO
        return x

    @classmethod
    def list_from_multipart(cls, multipart_media_type_obj: MediaTypeObject, stage): # -> list[MountPathFile]:
        '''
        Extract list of files to mount when multipart/form-data is used to
        describe a schema of multiple files.

        Create a MountPathFiles from the JSON schema used in this project for
        describing files and their paths.
        '''

        media_type = multipart_media_type_obj.media_type
        assert media_type == 'multipart/form-data', \
            f'Expected multipart/form-data media type, but got "{media_type}"'

        schema = multipart_media_type_obj.schema
        assert schema.type == SchemaType.OBJECT, 'Only object schemas supported'
        assert schema.properties, 'Expected properties for multipart schema'

        mounts = []
        encoding = multipart_media_type_obj.encoding
        for path, schema in get_supported_file_schemas(schema, encoding):
            media_type = encoding[path]['contentType']
            # NOTE: The other encoding field ('format') is not regarded here.
            mount = cls(path, media_type, stage)
            mounts.append(mount)

        return mounts

