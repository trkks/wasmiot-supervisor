"""
This is a module :)
"""

from datetime import datetime
from dataclasses import dataclass, field
import logging
import os
import socket
from pathlib import Path
import queue
import string
import struct
import threading
from typing import Any, Tuple

import atexit
from typing import Tuple
from flask import Flask, Blueprint, jsonify, current_app, request, send_file
from flask.helpers import get_debug_flag
from werkzeug.serving import get_sockaddr, select_address_family
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import secure_filename

import requests
from zeroconf import ServiceInfo, Zeroconf

import cv2
import numpy as np
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from host_app.wasm_utils.wasm import wasm_modules, wasm_runtime
from host_app.wasm_utils.wasm_api import MLModel, ModuleConfig

from host_app.utils.configuration import get_device_description, get_wot_td
from host_app.utils.routes import endpoint_failed
from host_app.utils.deployment import Deployment, CallData


_MODULE_DIRECTORY = 'wasm-modules'
_PARAMS_FOLDER = 'wasm-params'

OUTPUT_LENGTH_BYTES = 32 // 8
"""
Size in bytes of the length-type used to represent the size of the block of Wasm
memory containing output result of an executed WebAssembly function. e.g.
a block of 257 bytes can be enumerated with a 2 byte type such as a 16bit
integer but not with 1 byte.
"""

ALLOC_NAME = "alloc"
"""
Name used for the memory-allocation function that should be found in most every
WebAssembly module up for execution. Should have one parameter, which is the
length in bytes of the memory to allocate and returns beginning address of the
allocated block.
"""

@dataclass
class FetchFailures(Exception):
    """Raised when fetching modules or their attached files fails"""
    errors: list[requests.Response]


bp = Blueprint(os.environ["FLASK_APP"], os.environ["FLASK_APP"])

logger = logging.getLogger(os.environ["FLASK_APP"])

deployments = {}
"""
Mapping of deployment-IDs to instructions for forwarding function results to
other devices and calling their functions
"""

@dataclass
class RequestEntry():
    '''Describes a request of WebAssembly execution'''
    request_id: str = field(init=False)
    '''Unique identifier for this request'''
    deployment_id: str
    module_name: str
    function_name: str
    method: str
    request_args: Any
    request_files: list[str] # List of `Path`s
    work_queued_at: datetime
    result: Any = None

    def __post_init__(self):
        # TODO: Hash the ID (and include args and time as well) because in this
        # current way multiple same requests get overwritten.
        self.request_id = f'{self.deployment_id}:{self.module_name}:{self.function_name}'

request_history = []
'''Log of all the requests handled by this supervisor'''

wasm_queue = queue.Queue()
'''Queue of work for asynchoronous WebAssembly execution'''

def do_wasm_work(entry):
    '''
    Run a WebAssembly function and follow deployment instructions on what to
    do with its output.

    Return response of the possible call made or the raw result if chaining is
    not required.
    '''

    deployment = deployments[entry.deployment_id]

    try:
        print(f'Preparing Wasm module "{entry.module_name}"...')
        module, wasm_args, wasm_out_args = deployment.interpret_args_for(
            entry.module_name,
            entry.function_name,
            entry.request_args,
            entry.request_files
        )

        print(f'Running Wasm function "{entry.function_name}"...')
        raw_output = module.run_function(entry.function_name, wasm_args + wasm_out_args)
        print(f'Result: {raw_output}')
    except Exception as err:
        print(f"Error running WebAssembly function '{entry.function_name}':", err)
        return str(err)

    # Do the next call, passing chain along and return immediately (i.e. the
    # answer to current request should not be such, that it significantly blocks
    # the whole chain).
    this_result, next_call = deployment.interpret_call_from(
        module.name, entry.function_name, wasm_out_args, raw_output
    )

    if not isinstance(next_call, CallData):
        # No sub-calls needed.
        return this_result

    headers = next_call.headers
    files = next_call.files

    sub_response = getattr(requests, next_call.method)(
        next_call.url,
        timeout=10,
        files=files,
        headers=headers,
    )

    return sub_response.json()["resultUrl"]

def make_history(entry):
    '''Add entry to request history after executing its work'''
    entry.result = do_wasm_work(entry)
    request_history.append(entry)
    return entry

def wasm_worker():
    '''Constantly try dequeueing work for using WebAssembly modules'''
    while True:
        entry = wasm_queue.get()
        make_history(entry)
        wasm_queue.task_done()

# Turn-on the worker thread.
threading.Thread(target=wasm_worker, daemon=True).start()

def create_app(*args, **kwargs) -> Flask:
    '''
    Create a new Flask application.

    Registers the blueprint and initializes zeroconf.
    '''
    app = Flask(os.environ["FLASK_APP"], *args, **kwargs)

    # Create instance directory if it does not exist.
    Path(app.instance_path).mkdir(exist_ok=True)

    app.config.update({
        'secret_key': 'dev',
        'MODULE_FOLDER': Path(app.instance_path, _MODULE_DIRECTORY),
        'PARAMS_FOLDER': Path(app.instance_path, _PARAMS_FOLDER),
    })

    # Load config from instance/ -directory
    app.config.from_pyfile("config.py", silent=True)

    # add sentry logging
    app.config.setdefault('SENTRY_DSN', os.environ.get('SENTRY_DSN'))

    sentry_dsn = app.config.get("SENTRY_DSN")

    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[
                FlaskIntegration(),
                #RequestsIntegration(),
            ],
            # Set traces_sample_rate to 1.0 to capture 100%
            traces_sample_rate=1.0
        )
        print("Sentry logging is set up!")
    else:
        print("Sentry not configured")

    app.register_blueprint(bp)

    # If werkzeug reloader is running, it starts app in subprocess.
    # See: https://werkzeug.palletsprojects.com/en/2.1.x/serving/#reloader
    # To prevent broadcasting services when in main reloader thread,
    # try detecting if running on debug mode, and in reloader thread.
    # Todo: If reloaded is enabled, but debugging is not, this will fail.
    if not get_debug_flag() or is_running_from_reloader():
        init_zeroconf(app)

    return app


def init_zeroconf(app: Flask):
    """
    Initialize zeroconf service
    """
    server_name = app.config['SERVER_NAME'] or socket.gethostname()
    host, port = get_listening_address(app)

    properties={
        'path': '/',
        'tls': 1 if app.config.get("PREFERRED_URL_SCHEME") == "https" else 0,
    }

    service_info = ServiceInfo(
        type_='_webthing._tcp.local.',
        name=f"{app.name}._webthing._tcp.local.",
        addresses=[socket.inet_aton(host)],
        port=port,
        properties=properties,
        server=f"{server_name}.local.",
    )

    app.zeroconf = Zeroconf()
    app.zeroconf.register_service(service_info)

    atexit.register(teardown_zeroconf, app)


def teardown_zeroconf(app: Flask):
    """
    Stop advertising mdns services and tear down zeroconf.
    """
    try:
        app.zeroconf.generate_unregister_all_services()
    except TimeoutError:
        logger.debug("Timeout while unregistering mdns services, handling it gracefully.", exc_info=True)

    finally:
        app.zeroconf.close()


def get_listening_address(app: Flask) -> Tuple[str, int]:
    """
    Return the address Flask application is listening.

    TODO: Does not detect if listening on multiple addresses.
    """

    # Copied from flask/app.py and werkzeug.service how it determines address,
    # as serving address is not stored. By default flask uses request iformation
    # for this, but we can't rely on that.

    host = None
    port = None

    # Try guessing from server name, default path for flask.
    server_name = app.config.get("SERVER_NAME")
    if server_name:
        host, _, port = server_name.partition(":")
    port = port or app.config.get("PORT") or 5000

    # Fallback
    if not host:
        # From https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/28950776#28950776
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("10.255.255.255", 1))
            host, *_ = s.getsockname()
        except Exception:
            host = "0.0.0.0"
        finally:
            s.close()

    address_family = select_address_family(host, port)
    server_address = get_sockaddr(host, int(port), address_family)

    return server_address

@bp.route('/.well-known/wasmiot-device-description')
def wasmiot_device_description():
    '''Return the device description containing host functions in JSON'''
    return jsonify(get_device_description())

@bp.route('/.well-known/wot-thing-description')
def thingi_description():
    '''Return the Web of Things thing description in JSON'''
    return jsonify(get_wot_td())

@bp.route('/health')
def thingi_health():
    '''Return a report of the current health status of this thing'''
    import random
    return jsonify({
         "cpuUsage": random.random()
    })

def results_route(request_id=None, full=False):
    '''
    Return the route where execution/request results can be read from.

    If full is True, return the full URL, otherwise just the path. This is so
    that routes can easily return URL for caller to read execution results.
    '''
    root = 'request-history'
    route = f'{root}/{request_id}' if request_id else root
    return f'{request.root_url}{route}' if full else route

@bp.route('/' + results_route())
@bp.route('/' + results_route('<request_id>'))
def request_history_list(request_id=None):
    '''Return a list of or a specific entry result from previous call'''
    entry = next(
            (jsonify(x) for x in request_history if x.request_id == request_id),
            endpoint_failed(request, 'no matching entry in history', 404)
        ) \
        if request_id \
        else None

    return entry if entry else jsonify(request_history)

@bp.route('/<deployment_id>/modules/<module_name>/<function_name>', methods=["GET", "POST"])
def run_module_function(deployment_id, module_name, function_name):
    '''
    Execute the function in WebAssembly module and act based on instructions
    attached to the deployment of this call/execution.
    '''
    if not deployment_id in deployments:
        return endpoint_failed(request, 'deployment does not exist', 404)

    if module_name not in deployments[deployment_id].modules:
        return endpoint_failed(request, f"module {module_name} not found for this deployment")

    # Write input data to filesystem.
    input_file_paths = []
    if (input_data_file := request.files.get('data')):
        input_file_path = os.path.join(
            current_app.config['PARAMS_FOLDER'],
            input_data_file.filename
        )
        input_data_file.save(input_file_path)
        input_file_paths.append(Path(input_file_path))

    entry = RequestEntry(
        deployment_id,
        module_name,
        function_name,
        request.method,
        request.args,
        list(map(str, input_file_paths)),
        datetime.now()
    )

    # Assume that the work wont take long and do it synchronously on GET.
    if request.method.lower() == 'get':
        make_history(entry)
    else:
        # Send data to worker thread to handle non-blockingly.
        wasm_queue.put(entry)

    # Return a link to this request's result (which could link further until
    # some useful value is found).
    return jsonify({ 'resultUrl': results_route(entry.request_id, full=True) })

@bp.route('/modules/<module_name>/<function_name>' , methods=["POST"])
def run_module_function_raw_input(module_name, function_name):
    """
    Run a Wasm function from a module operating on Wasm-runtime's memory.

    The function's input arguments and output is passed and read by indexing
    into Wasm-runtime memory much like described in
    https://radu-matei.com/blog/practical-guide-to-wasm-memory/.

    The input is expected to be a byte-sequence found in `request.data`.
    """

    # Setup variables needed for initialization and running modules.
    module_config = wasm_modules.get(module_name, None)
    if not module_config or not function_name:
        return endpoint_failed(request, "not found")

    input_data = request.data

    # wu.load_module(module)

    # Allocate pointer to a suitable block of memory in Wasm and write the
    # input there.
    try:
        module = wasm_runtime.get_or_load_module(module_config)
        input_ptr = module.run_function(ALLOC_NAME, [len(input_data)])
    except Exception as err:
        return endpoint_failed(
            request,
            f"Failed running WebAssembly '{ALLOC_NAME}' for reserving {len(input_data)} bytes: {err}"
        )

    # Copy the input data into the allocated memory block.
    write_err = wasm_runtime.write_to_memory(input_ptr, input_data)
    if write_err is not None:
        return endpoint_failed(request, write_err)

    # Reserve memory for WebAssembly to write the length of the generated
    # output, so it can be read later and used in reading the _actual_ result.
    try:
        output_len_ptr = module.run_function(ALLOC_NAME, [OUTPUT_LENGTH_BYTES])
    except Exception as err:
        return endpoint_failed(
            request,
            f"Failed running WebAssembly '{ALLOC_NAME}' for reserving {OUTPUT_LENGTH_BYTES} bytes: {err}"
        )

    try:
        # NOTE: The parameters of the WebAssembly function being run is
        # constrained here. Expecting it to be:
        # Three (3) parameters:
        #   1) input buffer address
        #   2) length of input buffer
        #   3) address for writing output buffer's length
        # One output:
        #   - output buffer address
        input_params = [input_ptr, len(input_data), output_len_ptr]

        print(
            f"Running WebAssembly function '{function_name}' with params: ({', '.join((str(i) for i in input_params))})"
        )

        output_ptr = module.run_function(function_name, input_params)
    except Exception as err:
        return endpoint_failed(
            request,
            f"Failed running WebAssembly '{function_name}' with inputs ({', '.join((str(i) for i in input_params))}): {err}"
        )

    # Get the one unsigned int (4-byte) as little-endian like the Wasm memory
    # should be according to:
    # https://webassembly.org/docs/portability/
    output_len_data, read_err = wasm_runtime.read_from_memory(output_len_ptr, OUTPUT_LENGTH_BYTES, module_name)
    if read_err is not None:
        return endpoint_failed(request, read_err)

    try:
        # TODO Remove the hardcode somehow; size of the type is in the constant
        # OUTPUT_LENGTH_BYTES...Or could just agree on using 32 bits (unsigned int)
        # always.
        output_len = struct.unpack("<I", output_len_data)[0]
    except struct.error as err:
        return endpoint_failed(
            request,
            f"Interpreting WebAssembly output length failed: {err}"
        )

    print(f"Output result length is {output_len} bytes")

    # Read result from memory and pass forward TODO: Follow the deployment
    # sequence and instructions.
    output_data, read_err = wasm_runtime.read_from_memory(output_ptr, output_len, module_name)
    if read_err is not None:
        return endpoint_failed(request, read_err)

    try:
        # FIXME: Interpreting random byte sequence to string.
        result = output_data.decode("utf-8")
        if any(map(lambda c: c not in string.printable, result)):
            result = str(output_data)
    except struct.error as err:
        return endpoint_failed(
            request,
            f"Interpreting WebAssembly output result failed: {err}"
        )

    print(f"Result interpreted to string is '{result}'")

    return jsonify({ 'result': result })

@bp.route('/foo')
def serve_test_jpg():
    return send_file('temp_image.jpg')

@bp.route('/ml/<module_name>', methods=['POST'])
def run_ml_module(module_name = None):
    """Data for module comes as file 'data' in file attribute"""
    if not module_name:
        return jsonify({'status': 'error', 'result': 'module not found'})

    module_config = wasm_modules.get(module_name, None)
    if module_config is None:
        return endpoint_failed(request, f"module {module_name} not found")

    module = wasm_runtime.get_or_load_module(module_config)
    if module is None:
        return endpoint_failed(request, f"module {module_name} could not be loaded")

    file = request.files['data']
    if not file:
        return jsonify({'status': 'error', 'result': "file 'data' not in request"})
    data = file.read()

    res = module.run_ml_inference(module_config.ml_model, data)

    # TODO: remove the following direct response once the deployment id is included in the request
    return jsonify({ 'result': res })

    # TODO: Use a common error-handling function for all endpoints.
    try:
        # FIXME This very is ridiculous...
        resp_media_type, resp_schema = list(
            module_config.description['paths'][f'/ml/{{module}}']['post']['responses']['200']['content'].items()
        )[0]
        # TODO: Use the deployment-ID from the request.
        deployment_id = list(deployments.keys())[0]
        return deployments[deployment_id].call_chain(res, resp_media_type, resp_schema)
    except Exception as err:
        return endpoint_failed(request, str(err))

@bp.route('/ml/model/<module_name>', methods=['POST'])
def upload_ml_model(module_name = None):
    """Model comes as 'model' file in request file attribute"""
    if not module_name:
        return jsonify({'status': 'error', 'result': 'module not found'})

    file = request.files['model']
    if not file:
        return jsonify({'status': 'error', 'result': "file 'model' not in request"})

    path = Path(current_app.config['PARAMS_FOLDER']) / module_name / 'model'
    path.parent.mkdir(exist_ok=True, parents=True)
    file.save(path)

    model = MLModel(path)
    module_config = wasm_modules.get(module_name, None)
    if module_config is None:
        return endpoint_failed(request, f"module {module_name} not found")
    module_config.ml_model = model

    module = wasm_runtime.get_or_load_module(module_config)
    if module is None:
        return endpoint_failed(request, f"module {module_name} could not be loaded")

    return jsonify({'status': 'success'})

@bp.route('/img/<module_name>/<function_name>', methods=['POST'])
def run_img_function(module_name = None, function_name = None):
    """Image comes as a string of bytes (in file attribute)"""
    if not module_name or not function_name:
        return jsonify({'result': 'function of module not found'})
    module_config = wasm_modules.get(module_name, None)
    if module_config is None:
        return endpoint_failed(request, f"module {module_name} not found")
    module = wasm_runtime.get_or_load_module(module_config)
    if module is None:
        return endpoint_failed(request, f"module {module_name} could not be loaded")

    file = request.files['img']
    img = file.read()
    print(type(img))
    print(len(img))
    #file.save('image.png')
    #filebytes = np.fromstring(file.read(), np.uint8)
    #img = cv2.imdecode(filebytes, cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    shape = (480, 640, 3)
    #img_bytes = np.array(img).flatten().tobytes()
    img_bytes = img

    # FIXME: What is the correct value for data_ptr? And where should it be set?
    gs_img_bytes = module.run_data_function(
        function_name=function_name,
        data_ptr_function_name=module_config.data_ptr_function_name,
        data=img_bytes,
        params=[]
    )
    result = np.array(gs_img_bytes).reshape((shape))
    cv2.imwrite("../output/gsimg2.png", result)
    return jsonify({'status': 'success'})

@bp.route('/img2/<module_name>/<function_name>', methods=['POST'])
def run_grayscale(module_name = None, function_name = None):
    """Image comes as file"""
    if not module_name or not function_name:
        return jsonify({'result': 'function of module not found'})
    module_config = wasm_modules.get(module_name, None)
    if module_config is None:
        return endpoint_failed(request, f"module {module_name} not found")
    module = wasm_runtime.get_or_load_module(module_config)
    if module is None:
        return endpoint_failed(request, f"module {module_name} could not be loaded")

    file = request.files['img']
    #file.save('image.png')
    filebytes = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(filebytes, cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    shape = img.shape
    img_bytes = np.array(img).flatten().tobytes()

    # FIXME: What is the correct value for data_ptr? And where should it be set?
    gs_img_bytes = module.run_data_function(
        function_name=function_name,
        data_ptr_function_name=module_config.data_ptr_function_name,
        data=img_bytes,
        params=[]
    )
    result = np.array(gs_img_bytes).reshape((shape))
    cv2.imwrite("gsimg.png", result)
    return jsonify({'status': 'success'})

@bp.route('/deploy', methods=['POST'])
def get_deployment():
    """Parses the deployment from POST-request and enacts it. Request content-type needs to be 'application/json'"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'message': 'Non-existent or malformed deployment data'})
    modules = data['modules']

    if not modules:
        return jsonify({'message': 'No modules listed'})

    try:
        module_configs = fetch_modules(modules)
    except FetchFailures as err:
        print(err)
        return endpoint_failed(
            request,
            msg=f'{len(err.errors)} fetch failures',
            status_code=500,
            errors=err.errors
        )

    deployments[data["deploymentId"]] = Deployment(
        wasm_runtime,
        data["instructions"],
        module_configs,
    )

    # If the fetching did not fail (that is, crash), return success.
    return jsonify({'status': 'success'})

@bp.route('/upload_module', methods=['POST'])
def upload_module():
    if 'module' not in request.files:
        #flash('No module attached')
        return jsonify({'status': 'no module attached'})
    file = request.files['module']
    if file.filename.rsplit('.', 1)[1].lower() != 'wasm':
        return jsonify({'status': 'Only .wasm-files accepted'})
    filename = secure_filename(file.filename)
    file.save(os.path.join(current_app.config['MODULE_FOLDER'], filename))
    return jsonify({'status': 'success'})

@bp.route('/upload_params', methods=['POST'])
def upload_params():
    if 'params' not in request.files:
        return jsonify({'status': 'no params attached'})
    file = request.files['params']
    if file.filename.rsplit('.', 1)[1].lower() != 'json':
        return jsonify({'status': 'Only json-files accepted'})
    filename = secure_filename(file.filename)
    file.save(os.path.join(current_app.config['PARAMS_FOLDER'], filename))
    return jsonify({'status': 'success'})

def fetch_modules(modules) -> list[ModuleConfig]:
    """
    Fetch listed Wasm-modules, save them and their details and return data that
    can be used to instantiate modules for execution later.
    :modules: list of structs of modules to download
    """
    configs = []
    for module in modules:
        # Make all the requests at once.
        res_bin = requests.get(module["urls"]["binary"], timeout=5)
        res_desc = requests.get(module["urls"]["description"], timeout=5)
        res_others = [requests.get(x, timeout=5) for x in module["urls"]["other"]]

        # Check that each request succeeded before continuing on.
        # Gather errors together.
        errors = []
        for res in [res_bin, res_desc] + res_others:
            if not res.ok:
                errors.append(res)

        if errors:
            raise FetchFailures(errors)

        # "Request for module by name"
        module_path = os.path.join(current_app.config["MODULE_FOLDER"], module["name"])
        # Confirm that the module directory exists and create it if not TODO:
        # This would be better performed at startup.
        os.makedirs(current_app.config["MODULE_FOLDER"], exist_ok=True)
        with open(module_path, 'wb') as f:
            f.write(res_bin.content)

        # Add other listed files related to the module.
        data_files = []
        for res_other in res_others:
            # FIXME Assuming only one url and that it is for the model.
            other_path = Path(current_app.config['PARAMS_FOLDER']) / module['name'] / 'model'

            other_path.parent.mkdir(exist_ok=True, parents=True)
            with open(other_path, 'wb') as f:
                f.write(res_other.content)

            data_files.append(other_path)

            # update the module configuration with the model path
            # TODO: Does having a "model path" attribute in the module
            # config have direct benefits over "generic" files list?
            # new_module_config.ml_model = MLModel(other_path)

        # Save downloaded module's details.
        new_module_config = ModuleConfig(
            id=module["id"],
            name=module["name"],
            path=module_path,
            description=res_desc.json(),
            data_files=data_files,
        )
        configs.append(new_module_config)

    return configs
