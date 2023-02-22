import os
import atexit
import re
from typing import Tuple
from flask import Flask, Blueprint, jsonify, current_app, request
from flask.helpers import get_debug_flag
from werkzeug.serving import get_sockaddr, select_address_family
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import secure_filename
import logging

from zeroconf import ServiceInfo, Zeroconf
import socket

import cv2
import numpy as np

import wasm_utils.wasm_utils as wu
from utils.configuration import get_device_description, get_wot_td

MODULE_FOLDER = './modules'
PARAMS_FOLDER = './params'

bp = Blueprint('thingi', os.environ["FLASK_APP"])

logger = logging.getLogger(os.environ["FLASK_APP"])

def create_app(*args, **kwargs) -> Flask:
    """
    Create a new Flask application.

    Registers the blueprint and initializes zeroconf.
    """
    app = Flask(os.environ["FLASK_APP"], *args, **kwargs)

    app.config.update({
        'secret_key': 'dev',
        'MODULE_FOLDER': MODULE_FOLDER,
        'PARAMS_FOLDER': PARAMS_FOLDER,
    })

    # Load config from instance/ -directory
    app.config.from_pyfile("config.py", silent=True)

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