'''
Utilities for intepreting application deployments based on (OpenAPI) descriptions
of "things" (i.e., WebAssembly services/functions on devices) and executing
their instructions.
'''

from dataclasses import dataclass

from flask import jsonify
import requests


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
        Increment the program counter, wrapping around the amount of
        instructions if necessary.
        '''
        self.program_counter += 1
        self.program_counter %= len(self.instructions)

    def call_chain(self, wasm_result):
        '''
        Call a sequence of functions in order, passing the result of each to the
        next.

        Return the result of the recursive call chain or the local result which
        starts unwinding the chain.
        '''
        if self.program_counter > len(self.instructions):
            raise ProgramCounterExceeded(
                f'deployment sequence (length {len(self.instructions)}) exceeded (index {self.program_counter})'
            )

        target = self._next_target()

        if target is None:
            # Return the result back to caller, BEGINNING the unwinding of the
            # recursive requests.
            # TODO: Respond as was agreed upon in the OpenAPI description.
            return { 'ok': wasm_result }

        # Call next func in sequence based on its OpenAPI description.
        # FIXME: Assumes GET.
        # FIXME: Assumes path is already constructed and only "method parameters"
        # (i.e. "?foo=bar&baz=qux" in GET) need to be filled in.
        # FIXME: Uses the paths-dict like a list with one guaranteed item.
        target_path, path_obj = list(target['paths'].items())[0]

        # Figure out the method needed to call the next function and fill in the parameters.
        if (method := path_obj.get('get')) is not None:
            search = '?'
            for param in method['parameters']:
                search += f'{param["name"]}={wasm_result}&'
        elif (method := path_obj.get("post")) is not None:
            body = {}
            for param in method['parameters']:
                body[param['name']] = wasm_result
        else:
            raise NotImplementedError('Only GET and POST are supported.')

        target_url = target['servers'][0]['url'].rstrip("/") + '/' + target_path.lstrip("/") + search

        # Request to next node.
        # FIXME this way unnecessarily blocks execution although Flask probably
        # helps in part (handle by making an event-loop?).
        response = requests.get(target_url, timeout=5)

        # TODO: handle different response codes based on OpenAPI description.
        if response.status_code != 200:
            raise RequestFailed(f'Bad status code {response.status_code}')

        response_json = response.json()
        # TODO: Respond as was agreed upon in the OpenAPI description.
        return jsonify(response_json)
