import os
from pathlib import Path
import atexit
import re
from typing import Tuple
from flask import Flask, Blueprint, jsonify, current_app, request
from flask.helpers import get_debug_flag
from werkzeug.serving import get_sockaddr, select_address_family
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import secure_filename
import logging
import struct
import string

import requests
from zeroconf import ServiceInfo, Zeroconf
import socket

import cv2
import numpy as np
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

import wasm_utils.wasm_utils as wu
from utils.configuration import get_device_description, get_wot_td
from utils.routes import endpoint_failed
from utils.deployment import Deployment, ProgramCounterExceeded


MODULE_DIRECTORY = '../modules'
PARAMS_FOLDER = '../params'

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

bp = Blueprint('thingi', os.environ["FLASK_APP"])

logger = logging.getLogger(os.environ["FLASK_APP"])

from flask import Flask

deployments = {}
"""
Mapping of deployment-IDs to instructions for forwarding function results to
other devices and calling their functions
"""


def create_app(*args, **kwargs) -> Flask:
    """
    Create a new Flask application.

    Registers the blueprint and initializes zeroconf.
    """
    app = Flask(os.environ["FLASK_APP"], *args, **kwargs)

    app.config.update({
        'secret_key': 'dev',
        'MODULE_FOLDER': MODULE_DIRECTORY,
        'PARAMS_FOLDER': PARAMS_FOLDER,
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
    return jsonify(get_device_description())

@bp.route('/.well-known/wot-thing-description')
def thingi_description():
    return jsonify(get_wot_td())

@bp.route('/<deployment_id>/modules/<module_name>/<function_name>')
def run_module_function(deployment_id, module_name = None, function_name = None):
    if not module_name or not function_name:
        return jsonify({'result': 'not found'})

    # Error if deployment-ID (TODO Or other access-control) does not check out.
    if deployment_id != 'adhoc' and deployment_id not in deployments:
        return endpoint_failed(request, 'deployment does not exist', 404)

    #param = request.args.get('param', default=1, type=int)
    #params = request.args.getlist('param')
    module = wu.wasm_modules[module_name]
    wu.load_module(module)
    types = wu.get_arg_types(function_name)  # get argument types
    params = [t(arg) for arg, t in zip(request.args.values(), types)]  # get parameters from get request with given types TODO: use parameter names according to description.
    res = wu.run_function(function_name, params)

    # Return immediately if this request was purposefully made not in relation
    # to an existing deployment.
    if deployment_id == 'adhoc':
        return jsonify({ 'result': res })

    # TODO: Use a common error-handling function for all endpoints.
    try:
        # FIXME This very is ridiculous...
        resp_media_type, resp_schema = list(
            module.description['openapi']['paths'][f'/{{deployment}}/modules/{{module}}/{function_name}']['get']['responses']['200']['content'].items()
        )[0]
        return deployments[deployment_id].call_chain(res, resp_media_type, resp_schema)
    except ProgramCounterExceeded as err:
        return endpoint_failed(request, str(err))

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
    module = wu.wasm_modules.get(module_name, None)
    if not module or not function_name:
        return endpoint_failed(request, "not found")

    input_data = request.data

    wu.load_module(module)

    # Allocate pointer to a suitable block of memory in Wasm and write the
    # input there.
    try:
        input_ptr = wu.run_function("alloc", [len(input_data)])
    except Exception as err:
        return endpoint_failed(
            request,
            f"Failed running WebAssembly '{ALLOC_NAME}' for reserving {len(input_data)} bytes: {err}"
        )

    # Copy the input data into the allocated memory block.
    write_err = wu.write_to_memory(input_ptr, input_data)
    if write_err is not None:
        return endpoint_failed(request, write_err)

    # Reserve memory for WebAssembly to write the length of the generated
    # output, so it can be read later and used in reading the _actual_ result.
    try:
        output_len_ptr = wu.run_function("alloc", [OUTPUT_LENGTH_BYTES])
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

        output_ptr = wu.run_function(function_name, input_params)
    except Exception as err:
        return endpoint_failed(
            request,
            f"Failed running WebAssembly '{function_name}' with inputs ({', '.join((str(i) for i in input_params))}): {err}"
        )

    # Get the one unsigned int (4-byte) as little-endian like the Wasm memory
    # should be according to:
    # https://webassembly.org/docs/portability/
    output_len_data, read_err = wu.read_from_memory(output_len_ptr, OUTPUT_LENGTH_BYTES)
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
    output_data, read_err = wu.read_from_memory(output_ptr, output_len)
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


@bp.route('/ml/<module_name>', methods=['POST'])
def run_ml_module(module_name = None):
    """Data for module comes as file 'data' in file attribute"""
    if not module_name:
        return jsonify({'status': 'error', 'result': 'module not found'})

    module = wu.wasm_modules[module_name]
    # Re-initialize runtime to clear memory.
    wu.rt = wu.env.new_runtime(15000)
    wu.load_module(module)

    file = request.files['data']
    if not file:
        return jsonify({'status': 'error', 'result': "file 'data' not in request"})

    res = wu.run_ml_model(module_name, file) 

    # TODO: Use a common error-handling function for all endpoints.
    try:
        # FIXME This very is ridiculous...
        resp_media_type, resp_schema = list(
            module.description['openapi']['paths'][f'/ml/{{module}}']['post']['responses']['200']['content'].items()
        )[0]
        # TODO: Use the deployment-ID from the request.
        deployment_id = list(deployments.keys())[0]
        return deployments[deployment_id].call_chain(res, resp_media_type, resp_schema)
    except ProgramCounterExceeded as err:
        return endpoint_failed(request, str(err))
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
    
    path = wu.wasm_modules[module_name].model_path
    if not path:
        path = Path(current_app.config['PARAMS_FOLDER']) / module_name / 'model'
        wu.wasm_modules[module_name].model_path = path
    path.parent.mkdir(exist_ok=True, parents=True)
    file.save(path)
    return jsonify({'status': 'success'})

@bp.route('/img/<module_name>/<function_name>', methods=['POST'])
def run_img_function(module_name = None, function_name = None):
    """Image comes as a string of bytes (in file attribute)"""
    if not module_name or not function_name:
        return jsonify({'result': 'function of module not found'})
    wu.load_module(wu.wasm_modules[module_name])
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
    gs_img_bytes = wu.run_data_function(function_name, wu.wasm_modules[module_name].data_ptr, img_bytes)
    result = np.array(gs_img_bytes).reshape((shape))
    cv2.imwrite("../output/gsimg2.png", result)
    return jsonify({'status': 'success'})

@bp.route('/img2/<module_name>/<function_name>', methods=['POST'])
def run_grayscale(module_name = None, function_name = None):
    """Image comes as file"""
    if not module_name or not function_name:
        return jsonify({'result': 'function of module not found'})
    wu.load_module(wu.wasm_modules[module_name])
    file = request.files['img']
    #file.save('image.png')
    filebytes = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(filebytes, cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    shape = img.shape
    img_bytes = np.array(img).flatten().tobytes()
    gs_img_bytes = wu.run_data_function(function_name, wu.wasm_modules[module_name].data_ptr, img_bytes)
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
    deployments[data["deploymentId"]] = Deployment(data["instructions"])

    if not modules:
        return jsonify({'message': 'No modules listed'})

    try:
        fetch_modules(modules)
    except Exception as err:
        msg = f"Fetching modules failed: {err}"
        print(msg)
        return endpoint_failed(request, msg)

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

def fetch_modules(modules):
    """
    Fetch listed Wasm-modules and save them and their details.
    :modules: list of names of modules to download
    """
    for module in modules:
        res_bin = requests.get(module["urls"]["binary"])
        res_desc = requests.get(module["urls"]["description"])

        # Check that request succeeded before continuing on.
        # TODO: Could maybe request alternative sources from orchestrator for
        # getting this module?
        if not res_bin.ok or not res_desc.ok:
            raise Exception(f'Fetching module \'{module["name"]}\' from \'{module["url"]}\' failed: {res_bin.content or res_desc.content}')

        "Request for module by name"
        module_path = os.path.join(current_app.config["MODULE_FOLDER"], module["name"])
        # Confirm that the module directory exists and create it if not TODO:
        # This would be better performed at startup.
        os.makedirs(current_app.config["MODULE_FOLDER"], exist_ok=True)
        with open(module_path, 'wb') as f:
            f.write(res_bin.content)

        "Save downloaded module to module directory"
        wu.wasm_modules[module["name"]] = wu.WasmModule(
            name=module["name"],
            path=module_path,
            description=res_desc.json(),
        )
        "Add module details to module config"

        # Add other listed files related to the module.
        # FIXME Assuming only one url and that it is for the model.
        for other_url in module["urls"]["other"]:
            res_other = requests.get(other_url)
            if not res_other.ok:
                raise Exception(f'Fetching module \'{module["name"]}\' from \'{other_url}\' failed: {res_other.content}')
            otherPath = wu.wasm_modules[module['name']].model_path
            if not otherPath:
                otherPath = Path(current_app.config['PARAMS_FOLDER']) / module['name'] / 'model'
                wu.wasm_modules[module['name']].model_path = otherPath
            otherPath.parent.mkdir(exist_ok=True, parents=True)
            with open(otherPath, 'wb') as f:
                f.write(res_other.content)
