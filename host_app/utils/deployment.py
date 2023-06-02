'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

import json
import struct
from dataclasses import dataclass
from math import prod

import cv2
import numpy as np
import requests
from flask import jsonify

import wasm_utils.wasm_utils as wu


WASM_MEM_IMG_SHAPE = (480, 640, 3)

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
        self.program_counter += 1
        # DEBUG: Wrap to beginning.
        self.program_counter %= len(self.instructions)
        return target

    def call_chain(self, func_result, func_out_media_type, func_out_schema):
        '''
        Call a sequence of functions in order, passing the result of each to the
        next.

        Return the result of the recursive call chain or the local result which
        starts unwinding the chain.
        '''

        # From the WebAssembly function's execution, parse result into the type
        # that needs to be used as argument for next call in sequence.
        parsed_result = parse_func_result(func_result, func_out_media_type, func_out_schema)

        # Select whether to forward the result to next node (deepening the call
        # chain) or return it to caller (respond).
        target = self._next_target()
        sub_request_is_needed = target is not None

        if sub_request_is_needed:
            return handle_sub_request(target, parsed_result, func_out_media_type)
        else:
            return return_immediately(parsed_result, func_out_media_type)
             
def parse_func_result(func_result, expected_media_type, expected_schema):
    '''
    Interpret the result of a function call based on the function's OpenAPI
    description.
    '''
    # DEMO: This is how the Camera service is invoked (no input).
    if expected_media_type == 'application/json':
        # TODO: For other than 'null' JSON, parse object from func_result (which
        # might be a (fat)pointer to Wasm memory).
        response_obj = None
    # DEMO: This is how the ML service is sent an image.
    elif expected_media_type == 'image/jpeg':
        # Read the constant sized image from memory.
        # FIXME Assuming there is this function that gives the buffer address
        # found in the module.
        img_address = wu.rt.find_function('get_img_ptr')()
        # FIXME Assuming the buffer size is according to this constant
        # shape.
        data_len = prod(WASM_MEM_IMG_SHAPE)
        img_bytes, err = wu.read_from_memory(img_address, data_len, to_list=True)
        if err:
            raise RuntimeError(f'Could not read image from memory: {err}')
        # Store raw bytes for now.
        response_obj = img_bytes
    # DEMO: This how the Camera service receives back the classification result.
    elif expected_media_type == 'application/octet-stream':
        response_obj = func_result
    else:
        raise NotImplementedError(f'Unsupported response media type {expected_media_type}')

    return response_obj

def request_to(url, media_type, payload):
    """
    Make a (sub or 'recursive') POST request to a URL selecting the placing of
    payload from media type.

    :return Response from `requests.post`
    """
    # List of key-path-mode -tuples for reading files on request.
    files = []
    data = None
    headers = {}
    if media_type == 'application/json' or \
        media_type == 'application/octet-stream':
        # HACK
        headers = { "Content-Type": media_type }
        data = payload
    elif media_type == 'image/jpeg':
        TEMP_IMAGE_PATH = 'temp_image.jpg'
        # NOTE: 'payload' at this point expected to be raw bytes read from
        # memory.

        img = np.array(payload).reshape(WASM_MEM_IMG_SHAPE)
        cv2.imwrite(TEMP_IMAGE_PATH, img)

        # TODO: Is this 'data' key hardcoded into ML-path and should it
        # instead be in an OpenAPI doc?
        files.append(("data", TEMP_IMAGE_PATH, "rb"))
    else:
        raise NotImplementedError(f'bug: media type unhandled "{media_type}"')

    files = { key: open(path, mode) for (key, path, mode) in files }

    return requests.post(
        url,
        timeout=120,
        data=data,
        files=files,
        headers=headers,
    )

def handle_sub_request(target, input_value, input_type):
    """
    Call the target endpoint with input.

    :return Response compatible with a Flask route's response (i.e., JSON, text etc.).
    """

    target_path, target_path_obj = list(target['paths'].items())[0]

    target_url = target['servers'][0]['url'].rstrip("/") + '/' + target_path.lstrip("/")

    # Request to next node.
    # NOTE: This makes a blocking call.
    response = None
    # Fill in the parameters according to call method.
    if 'post' in target_path_obj:
        response = request_to(target_url, input_type, input_value)
    else:
        raise NotImplementedError('Only POST is supported but was not found in target endpoint description.')

    # TODO: handle different response codes based on OpenAPI description.
    if response.status_code != 200:
        raise RequestFailed(f'Bad status code {response.status_code}')
    
    # From:
    # https://stackoverflow.com/questions/19568950/return-a-requests-response-object-from-flask
    return response.content, response.status_code, response.headers.items()

def return_immediately(output_value, output_type):
    """

    :return Response compatible with a Flask route's response (i.e., JSON, text etc.).
    """
    # Return the result back to caller, BEGINNING the unwinding of the
    # recursive requests.
    if output_type == 'application/octet-stream':
        # TODO: Technically this should just return the bytes but figuring
        # that out seems too much of a hassle right now...
        bytes = [b for b in struct.pack("<I", output_value)]
        return bytes
    elif output_type == "application/json":
        return jsonify(json.loads(output_value))
    else:
        raise NotImplementedError(f'bug: media type unhandled "{output_type}"')
