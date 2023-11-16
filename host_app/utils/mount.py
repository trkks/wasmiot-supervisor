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

@dataclass
class MountPathFile:
    '''
    Defines the schema used for files in "multipart/form-data" requests
    '''
    path: str
    media_type: str
    stage: MountStage | str
    required = True
    encoding: str = 'base64'
    type: str = 'string'

    def __post_init__(self):
        """Initialize the other dataclass fields"""
        if isinstance(self.stage, str):
            self.stage = MountStage(self.stage)

    @classmethod
    def validate(cls, x: dict[str, Any]): # -> MountPathFile:
        # TODO
        return x