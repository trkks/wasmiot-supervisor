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

MODULE_DIRECTORY = '../modules'
PARAMS_FOLDER = '../params'

PTR_BYTES = 32 // 8
"Size in bytes of the pointer used to index Wasm memory."
LENGTH_BYTES = 32 // 8
"""
Size in bytes of the length-type used to represent a Wasm memory block size e.g.
a block of 257 bytes can be enumerated with 2 bytes but not with 1 byte.
"""

bp = Blueprint('thingi', os.environ["FLASK_APP"])

logger = logging.getLogger(os.environ["FLASK_APP"])

from flask import Flask


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

@bp.route('/modules/<module_name>/<function_name>')
def run_module_function(module_name = None, function_name = None):
    if not module_name or not function_name:
        return jsonify({'result': 'not found'})
    #param = request.args.get('param', default=1, type=int)
    #params = request.args.getlist('param')
    wu.load_module(wu.wasm_modules[module_name])
    types = wu.get_arg_types(function_name)  # get argument types
    params = [request.args.get('param' + str(i+1), type=t) for i, t in enumerate(types)]  # get parameters from get request (named param1, param2, etc.) with given types
    res = wu.run_function(function_name, params)
    return jsonify({'result': res})


@bp.route('/modules/<module_name>/<function_name>' , methods=["POST"])
def run_module_function_raw_input(module_name, function_name):
    """
    Run a Wasm function from a module operating on Wasm-runtime's memory.
    
    The function's input arguments and output is passed and read by indexing
    into Wasm-runtime memory much like described in
    https://radu-matei.com/blog/practical-guide-to-wasm-memory/.

    The input is expected to be a byte-sequence found in `request.data`.
    """
    # FIXME: This route seems non-functional in my experiments and needs more
    # work if raw bytes input/output is required in the future. So until it
    # gets functional, exit immediately.
    return endpoint_failed(request, "Byte input route not supported")

    # Setup variables needed for initialization and running modules.
    module = wu.wasm_modules.get(module_name, None)
    if not module or not function_name:
        return endpoint_failed(request, "not found")

    input_data = request.data
    input_len = len(input_data)

    # Allocate pointer to a suitable block of memory in Wasm and write the
    # input there.
    wu.load_module(module)

    try:
        ptr = wu.run_function("alloc", [input_len])
    except Exception as err:
        return endpoint_failed(request, f"Input buffer allocation failed: {err}")

    memory = wu.rt.get_memory(0)
    memory[ptr:ptr + input_len] = input_data

    # Run the application func now that input is written to its place.
    # The function naturally needs to be implemented to:
    # 1. receive the input pointer and length as arguments
    # 2. return the result pointer and length as output TODO This latter might
    # be iffy with some source-languages (can e.g. C be compiled to Wasm
    # returning tuples?)
    try:
        # NOTE: For functions, that return tuples like f() -> (ptr, len),
        # Wasm-compilers apparently write the result into the last parameter
        # (i.e. a memory address) for runtime-compatibility -reasons. See:
        # https://stackoverflow.com/questions/70641080/wasm-from-rust-not-returning-the-expected-types
        import struct
        # Allocate space for the tuple return value (which then points to
        # _application_ return value).
        try:
            ret_ptr = wu.run_function("alloc", [PTR_BYTES + LENGTH_BYTES])
        except Exception as err:
            return endpoint_failed(request, f"Output buffer allocation failed: {err}")

        wu.run_function(function_name, [ptr, input_len, ret_ptr])
        # The pointer to _application_ result should be found in memory with
        # address first and length second.
        ret_slice = memory[ret_ptr:ret_ptr + PTR_BYTES + LENGTH_BYTES]

        # Here's the docstring of tobytes from the python interpreter:
        # >>> ret_slice.tobytes.__doc__
        # "Return the data in the buffer as a byte string.\n\nOrder can be {'C',
        # 'F', 'A'}. When order is 'C' or 'F', the data of the\noriginal array
        # is converted to C or Fortran order. For contiguous views,\n'A' returns
        # an exact copy of the physical memory. In particular,
        # in-memory\nFortran order is preserved. For non-contiguous views, the
        # data is converted\nto C first. order=None is the same as order='C'."
        ret_bytes = ret_slice.tobytes(order="A")
        # Get the two (2) unsigned ints (4-byte) (NOTE Hardcoded although sizes
        # of the types are in the constants as used above) as little-endian like
        # the Wasm memory should be according to:
        # https://webassembly.org/docs/portability/
        res_ptr, res_len = struct.unpack("<II", ret_bytes)
    except Exception as err:
        import traceback
        traceback.print_exc()
        return endpoint_failed(request, f"Execution failed: {err}")

    print("Result address and length:", res_ptr, res_len)

    # Read result from memory and pass forward TODO: Follow the deployment
    # sequence and instructions.
    res = memory[res_ptr:res_ptr + res_len]

    # FIXME: Interpreting random byte sequence to string.
    return jsonify({'result': str(res)})


@bp.route('/ml/<module_name>', methods=['POST'])
def run_ml_module(module_name = None):
    """Data for module comes as file 'data' in file attribute"""
    if not module_name:
        return jsonify({'status': 'error', 'result': 'module not found'})

    wu.load_module(wu.wasm_modules[module_name])

    file = request.files['data']
    if not file:
        return jsonify({'status': 'error', 'result': "file 'data' not in request"})

    res = wu.run_ml_model(module_name, file) 
    return jsonify({'status': 'success', 'result': res})

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
        r = requests.get(module["url"])

        # Check that request succeeded before continuing on.
        # TODO: Could maybe request alternative sources from orchestrator for
        # getting this module?
        if not r.ok:
            raise Exception(f'Fetching module \'{module["name"]}\' from \'{module["url"]}\' failed: {r.content}')

        "Request for module by name"
        module_path = os.path.join(current_app.config["MODULE_FOLDER"], module["name"])
        # Confirm that the module directory exists and create it if not TODO:
        # This would be better performed at startup.
        os.makedirs(current_app.config["MODULE_FOLDER"], exist_ok=True)
        open(module_path, 'wb').write(r.content)
        "Save downloaded module to module directory"
        wu.wasm_modules[module["name"]] = wu.WasmModule(
            name=module["name"],
            path=module_path,
        )
        "Add module details to module config"
