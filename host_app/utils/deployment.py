'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass
from math import prod

import cv2
from flask import jsonify, send_file
import numpy as np
import requests

import wasm_utils.wasm_utils as wu

class ProgramCounterExceeded(Exception):
    '''Raised when a deployment sequence is exceeded.'''

class RequestFailed(Exception):
    '''Raised when a chained request to a thing fails.'''


@dataclass
class Deployment:
    '''Describing a sequence of instructions to be executed in (some) order.'''
    instructions: list
    program_counter: int = 0

    def _next_target(self):
        '''
        Choose the next instruction's target and increment internal state to
        prepare for the next call.
        '''
        target = self.instructions[self.program_counter]['to']
        # Update the sequence ready for next call to this deployment.
        self._increment_program_counter()
        return target

    def _increment_program_counter(self):
        '''
        Set `self` to point to the next instruction in the sequence.
        '''
        self.program_counter += 1

        # TEMPORARY FOR DEBUG: wrap around to beginning of sequence.
        self.program_counter %= len(self.instructions)

        if self.program_counter >= len(self.instructions):
            raise ProgramCounterExceeded(
                f'deployment sequence (length {len(self.instructions)}) exceeded (index {self.program_counter})'
            )

    def call_chain(self, func_result, expected_media_type, expected_schema):
        '''
        Call a sequence of functions in order, passing the result of each to the
        next.

        Return the result of the recursive call chain or the local result which
        starts unwinding the chain.
        '''

        # From the WebAssembly function's execution, parse result into the type
        # that needs to be used as argument for next call in sequence.
        response_obj = parse_func_result(func_result, expected_media_type, expected_schema)

        # Select whether to forward the result to next node (deepening the call
        # chain) or return it to caller (respond).
        target = self._next_target()

        if target is not None:
            # Call next func in sequence based on its OpenAPI description.
            target_path, target_path_obj = list(target['paths'].items())[0]

            target_url = target['servers'][0]['url'].rstrip("/") + '/' + target_path.lstrip("/")

            # Request to next node.
            # NOTE: This makes a blocking call.
            sub_response = None
            # Fill in the parameters according to call method.
            if 'post' in target_path_obj:
                if expected_media_type in {'application/json', 'application/octet-stream'} and isinstance(response_obj, dict):
                    sub_response = requests.post(target_url, timeout=60, data=response_obj)
                elif expected_media_type == 'image/jpeg':
                    TEMP_IMAGE_PATH = 'temp_image.jpg'
                    cv2.imwrite(TEMP_IMAGE_PATH, response_obj)
                    with open(TEMP_IMAGE_PATH, 'rb') as f:
                        sub_response = requests.post(target_url, timeout=60, files={ "data": f })
                else:
                    raise NotImplementedError(f'bug: media type unhandled "{expected_media_type}"')
            else:
                raise NotImplementedError('Only POST is supported but was not found in target endpoint description.')

            # TODO: handle different response codes based on OpenAPI description.
            if sub_response.status_code != 200:
                raise RequestFailed(f'Bad status code {sub_response.status_code}')

            response_obj = sub_response.content
            expected_media_type = sub_response.headers['Content-Type']

        # Return the result back to caller, BEGINNING the unwinding of the
        # recursive requests.
        if expected_media_type == 'application/json':
            return jsonify(str(response_obj))
        elif expected_media_type == 'image/jpeg':
            TEMP_IMAGE_PATH = 'temp_image.jpg'
            cv2.imwrite(TEMP_IMAGE_PATH, response_obj)
            return send_file(TEMP_IMAGE_PATH)
        elif expected_media_type == 'application/octet-stream':
            # TODO: Technically this should just return the bytes...
            return jsonify({ 'result': response_obj })
        else:
            raise NotImplementedError(f'bug: media type unhandled "{expected_media_type}"')
     
def parse_func_result(func_result, expected_media_type, expected_schema):
    '''
    Interpret the result of a function call based on the function's OpenAPI
    description.
    '''
    # DEMO: This is how the Camera service is invoked (no input).
    if expected_media_type == 'application/json':
        response_obj = None
    # DEMO: This is how the ML service is sent an image.
    elif expected_media_type == 'image/jpeg':
        # Read the constant sized image from memory.
        # FIXME Assuming this function giving the buffer address is found in
        # the module.
        img_address = wu.run_function('get_img_ptr', b'')
        # FIXME Assuming the buffer size is according to this constant
        # shape.
        img_shape = (480, 640, 3)
        img_bytes, err = wu.read_from_memory(img_address, prod(img_shape), to_list=True)
        if err:
            raise RuntimeError(f'Could not read image from memory: {err}')
        response_obj = np.array(img_bytes).reshape(img_shape)
    # DEMO: This how the Camera service receives back the classification result.
    elif expected_media_type == 'application/octet-stream':
        response_obj = func_result
    else:
        raise NotImplementedError(f'Unsupported response media type {expected_media_type}')

    return response_obj 

