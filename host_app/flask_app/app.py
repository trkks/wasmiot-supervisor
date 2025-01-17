"""
This is a module :)
"""

from datetime import datetime
from dataclasses import dataclass, field
import itertools
import logging
import os
import random
import socket
from pathlib import Path
import queue
import threading
from typing import Any, Dict, Generator, Tuple
import time

import atexit
from flask import Flask, Blueprint, jsonify, current_app, request, send_file, url_for, Response
from werkzeug.serving import get_sockaddr, select_address_family
from werkzeug.serving import is_running_from_reloader
# TODO: Use this whenever doing file I/O:
from werkzeug.utils import secure_filename

import requests
from zeroconf import ServiceInfo, Zeroconf

from host_app.wasm_utils.wasm_api import ModuleConfig
from host_app.wasm_utils.wasmtime import WasmtimeRuntime

from host_app.utils.configuration import get_device_description, get_wot_td
from host_app.utils.routes import endpoint_failed
from host_app.utils.deployment import Deployment
from host_app.utils.mount import MountPathFile
from host_app.utils.call import CallData, call_endpoint


SIM_MAX_LATENCY_SECS = None
'''Set some max-latency of selected requests for simulating physical environment'''
try:
    SIM_MAX_LATENCY_SECS = float(os.environ.get('SIMULATED_LATENCY'))
except:
    pass

_MODULE_DIRECTORY = 'wasm-modules'
_PARAMS_FOLDER = 'wasm-params'
_OUTPUT_FOLDER = "wasm-output-files"
INSTANCE_PARAMS_FOLDER = None
INSTANCE_OUTPUT_FOLDER = None

@dataclass
class FetchFailures(Exception):
    """Raised when fetching modules or their attached files fails"""
    errors: list[requests.Response]

FLASK_APP = os.environ.get("FLASK_APP", __name__)

bp = Blueprint(os.environ["FLASK_APP"], os.environ["FLASK_APP"])

logger = logging.getLogger(FLASK_APP)

deployments = {}
"""
Mapping of deployment-IDs to instructions for forwarding function results to
other devices and calling their functions
"""
prepared_deployments = {}
'''
Mapping of deployment-IDs to ready-to-use -instructions that are not yet put to
use.
'''
deployment_locks: dict[str, threading.Lock] = {}
'''
Mapping of deployment-IDs to per-deployment locks that are used to block
requests using the deployment.
'''


def request_counter() -> Generator[int, None, None]:
    """Returns a unique number for each request"""
    counter: int = 0
    while True:
        counter += 1
        yield counter

request_id_counters: Dict[str, Generator[int, None, None]] = {}


def path_to_link(result: Any, request_id=None) -> Any:
    '''
    Converts all included output-mount paths (represented as strings) to links
    where the file is available based on request entry ID.
    '''
    if isinstance(result, MountPathFile):
        return results_route(request_id, full=True, file=result.path)
    if isinstance(result, (list, tuple)):
        return [path_to_link(item, request_id) for item in result]
    if isinstance(result, dict):
        return {key: path_to_link(value, request_id) for key, value in result.items()}
    if isinstance(result, RequestEntry):
        result.result = path_to_link(result.result, result.request_id)
    return result


def next_request_id(deployment_id, module_name, function_name):
    '''Generate a unique ID for a request'''
    # TODO: Hash the ID (and include args and time as well) because in this
    # current way multiple same requests get overwritten.
    # - For now a simple request counter is used to distinguish requests for the same function
    request_id = f'{deployment_id}:{module_name}:{function_name}'
    if request_id not in request_id_counters:
        request_id_counters[request_id] = request_counter()
    request_id = f'{request_id}:{next(request_id_counters[request_id])}'
    return request_id


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
    request_files: Dict[str, str]
    work_queued_at: datetime
    result: Any = None
    success: bool = False
    redirected: bool = False

    def __post_init__(self):
        self.request_id = next_request_id(self.deployment_id, self.module_name, self.function_name)
        

request_history: list[RequestEntry] = []
'''Log of all the requests handled by this supervisor'''

wasm_queue = queue.Queue()
'''Queue of work for asynchronous WebAssembly execution'''


def module_mount_path(module_name: str, filename: str | None = None) -> Path:
    """
    Return path for a file that will eventually be made available for a
    module
    """
    return Path(INSTANCE_PARAMS_FOLDER, module_name, filename if filename else "")

def per_request_file_path(request_id, mount_name):
    '''
    Return the file path where output-mounts are saved identified by request.
    '''
    return Path(INSTANCE_OUTPUT_FOLDER, request_id, mount_name)

def do_wasm_work(entry: RequestEntry):
    '''
    Run a WebAssembly function and follow deployment instructions on what to
    do with its output.

    Return response of the possible call made or the raw result if chaining is
    not required.
    '''

    if entry.deployment_id in deployment_locks:
        # Wait for a possible redeployment to finish and release the deployment
        # lock immediately after. Kinda like "barrier"-synchronization.
        dlock = deployment_locks[entry.deployment_id]
        logger.debug("Acquiring deployment lock for %r", entry.deployment_id)
        dlock.acquire()
        logger.debug("Releasing deployment lock for %r", entry.deployment_id)
        dlock.release()

    deployment = deployments[entry.deployment_id]

    logger.debug("Preparing Wasm module %r", entry.module_name)
    module, wasm_args = deployment.prepare_for_running(
        entry.module_name,
        entry.function_name,
        entry.request_args,
        entry.request_files
    )

    # Force calls to this module to wait.
    if not hasattr(module, 'lock'):
        module.lock = threading.Lock()
    logger.debug("Acquiring module lock for %r", module.name)
    module.lock.acquire()

    logger.debug("Running Wasm function %r", entry.function_name)
    raw_output = module.run_function(entry.function_name, wasm_args)
    logger.debug("... Result: %r", raw_output, extra={"raw_output": raw_output})

    # Release the module lock immediately after exec.
    logger.debug("Releasing module lock for %r", module.name)
    module.lock.release()

    # Do the next call, passing chain along and return immediately (i.e. the
    # answer to current request should not be such, that it significantly blocks
    # the whole chain).
    this_result, next_call = deployment.interpret_call_from(
        module.name, entry.function_name, raw_output
    )
 
    if not isinstance(next_call, CallData):
        # No sub-calls needed.
        # For any output mounts, save them to request-identifiable files.
        endpoint_data = this_result[1] or []
        for output_mount in endpoint_data:
            output_mount_path = module_mount_path(entry.module_name, output_mount.path)

            # NOTE: Only the mount filename is regarded i.e. mounts cannot have
            # directory structure!
            request_entry_path = per_request_file_path(entry.request_id, output_mount_path.name)
            
            # Make sure the directories exist before writing.
            if not request_entry_path.exists():
                os.makedirs(os.path.dirname(request_entry_path), exist_ok=True)

            with open(output_mount_path, "rb") as source:
                contents = source.read()
            logger.info("Mount %s contents length: %d", output_mount_path, len(contents))

            with open(request_entry_path, "wb") as target:
                target.write(contents)
            logger.info("Output file %s written", request_entry_path)

            # Clear the mount file now that is has been identifiably saved elsewhere.
            with open(output_mount_path, "wb") as _:
                pass

        return this_result

    sub_resp = call_endpoint(next_call, entry.module_name)

    return sub_resp.json()["resultUrl"]


def exec_log(entry: RequestEntry):
    '''
    Execute work described by the entry and add its result to the request
    history (i.e., the "log").
    '''
    try:
        entry.result = do_wasm_work(entry)
        entry.success = True
    except Exception as err:
        logger.error("Error running WebAssembly function %r", entry.function_name, exc_info=True, extra={
            "request": entry,
        })
        entry.result = str(err)
        entry.success = False

    request_history.append(entry)

def wasm_worker():
    '''Constantly try dequeueing work for using WebAssembly modules'''
    while entry := wasm_queue.get():
        exec_log(entry)
        wasm_queue.task_done()

def create_app(*args, **kwargs) -> Flask:
    '''
    Create a new Flask application.

    Registers the blueprint and initializes zeroconf.
    '''
    if is_running_from_reloader():
        raise RuntimeError("Running from reloader is not supported.")
    
    app = Flask(os.environ.get("FLASK_APP", __name__), *args, **kwargs)

    # Create instance directory if it does not exist.
    Path(app.instance_path).mkdir(exist_ok=True)

    app.config.update({
        'secret_key': 'dev',
        'MODULE_FOLDER': Path(app.instance_path, _MODULE_DIRECTORY),
        'PARAMS_FOLDER': Path(app.instance_path, _PARAMS_FOLDER),
        'OUTPUT_FOLDER': Path(app.instance_path, _OUTPUT_FOLDER),
    })

    # Set these in order to later access module params and outputs folder that
    # Flask has set up on app creation.
    global INSTANCE_PARAMS_FOLDER
    INSTANCE_PARAMS_FOLDER = app.config['PARAMS_FOLDER']
    global INSTANCE_OUTPUT_FOLDER
    INSTANCE_OUTPUT_FOLDER = app.config['OUTPUT_FOLDER']

    # Load config from instance/ -directory
    app.config.from_pyfile("config.py", silent=True)

    # add sentry logging
    app.config.setdefault('SENTRY_DSN', os.environ.get('SENTRY_DSN'))

    from .logging.logger import init_app as init_logging  # pylint: disable=import-outside-toplevel
    init_logging(app, logger=logger)

    app.register_blueprint(bp)

    # Enable mDNS advertising.
    init_zeroconf(app)

    # Start thread that handles the Wasm work queue.
    init_wasm_worker()

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


def init_wasm_worker():
    """
    Set up and start a thread that continuously dequeues given work for running
    Wasm.
    """
    def teardown_worker():
        """Signal the worker thread to stop and wait for it to finish."""
        wasm_queue.put(None)
        logger.debug("Waiting for the worker thread to finish...")
        wasm_worker_thread.join()
        logger.debug("Worker thread finished!")

    # Turn-on the worker thread.
    wasm_worker_thread = threading.Thread(target=wasm_worker, daemon=True)
    wasm_worker_thread.start()

    # Stop the worker thread before exiting.
    atexit.register(teardown_worker)


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
    # as serving address is not stored. By default flask uses request information
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
    if isinstance(server_address, str):
        # TODO: check if this case can happen and what should be the default port
        return server_address, 80
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
    return jsonify({
         "cpuUsage": random.random()
    })


def find_request(request_id):
    '''
    Return the single entry matched to the ID or None if not found.
    '''
    return next(filter(lambda x: x.request_id == request_id, request_history), None)


def results_route(request_id=None, *, full=False, file=None):
    '''
    Return the route where execution/request results can be read from.

    If full is True, return the full URL, otherwise just the path. This is so
    that routes can easily return URL for caller to read execution results.

    If the result should be a main script, return the route that serves said
    script relative to "deployment root".
    '''
    # NOTE: Handling main script execution here to return special URL that is
    # more ergonomically relative to Wasm execution route.
    if request_id \
        and full \
        and file is None \
        and (entry := find_request(request_id)) \
    :
        did = entry.deployment_id
        dep = deployments[did]
        is_main_script_execution = \
            dep.instructions.main == entry.module_name \
            and dep.instructions.start == entry.function_name
        if is_main_script_execution:
            return request.root_url.removesuffix('/') \
                + url_for(
                    f'{FLASK_APP}.deployment_index',
                    deployment_id=did, request_id=request_id,
                )

    root = 'request-history'
    base_route = f'{root}/{request_id}' if request_id else root
    if file:
        # HACKY: The glob symbol means all files, otherwise interpreted as file
        # name.
        base_route += '/files' if file == '*' else f'/files/{file}'
    return f'{request.root_url}{base_route}' if full else base_route


@bp.route('/' + results_route('<request_id>', file='*'))
@bp.route('/' + results_route('<request_id>', file='<filename>'))
def request_history_file_list(request_id, filename=None):
    '''Return a list of or a specific entry result __files__ of a previous execution call'''
    if filename is None:
        # TODO: Zip all the files for sending
        raise NotImplementedError

    return send_file(per_request_file_path(request_id, filename))

@bp.route('/' + results_route())
@bp.route('/' + results_route('<request_id>'))
def request_history_list(request_id=None):
    '''Return a list of or a specific entry result of a previous execution call'''

    if SIM_MAX_LATENCY_SECS is not None:
        waitsecs = random.random() * SIM_MAX_LATENCY_SECS
        logger.info('%f second simulated latency on request history route', waitsecs)
        time.sleep(waitsecs)

    if request_id is None:
        return jsonify(path_to_link(request_history))
    if not (match := find_request(request_id)):
        return endpoint_failed(request, 'no matching entry in history', 404)
    json_response = jsonify(path_to_link(match))
    json_response.status_code = 200 if match.success else 500
    return json_response


def name_is_local(deployment_id, module_name, function_name=None):
    '''Return True if the namepath exists locally.'''
    return deployments[deployment_id] \
        .local_endpoint(module_name, function_name) \
        is not None


def peer_has_module(deployment_id, module_name):
    '''Return True if some peer in deployment has the module.'''
    return deployment_id in deployments \
        and module_name in deployments[deployment_id].peers


@bp.route('/<deployment_id>/')
def deployment_index(deployment_id):
    '''
    If there is one, serve the deployment index page previously generated by a
    Wasm call.

    NOTE that on the ending "/" on the route is crucial for providing relative
    paths to other module functions i.e., "./modules/mymodule/myfunc".
    '''
    # Request id must be supplied in query params.
    request_id = request.args.get("request_id")

    # Find the result of previously ran index function that should have produced
    # the file of the index page.
    entry = find_request(request_id)

    if not entry:
        return endpoint_failed(request, 'request entry does not exist', 404)

    if entry.deployment_id != deployment_id:
        return endpoint_failed(request, 'request entry does not belong to the deployment', 403)

    if not entry.success:
        return endpoint_failed(request, f'index function had failed: "{entry.result}"', 500)

    if entry.result[1] is not None:
        output_mount = entry.result[1][0]
        output_mount_path = module_mount_path(entry.module_name, output_mount.path)
        request_entry_path = per_request_file_path(entry.request_id, output_mount_path.name)

        # Respond with the (assumed) index page.
        return send_file(request_entry_path)

    # Respond with the result of execution.
    return jsonify(entry.result[0])


def prepare_request_entry(deployment_id, module_name, function_name):
    '''Prepare inputs and a request entry based on current request and given identifiers'''
    # Write input data to filesystem.
    input_file_paths: Dict[str, str] = {}
    for param_name, input_data_file in request.files.items():
        input_file_path = os.path.join(
            current_app.config['PARAMS_FOLDER'],
            (
                input_data_file.name
                if input_data_file is not None and input_data_file.name is not None
                else ""
            )
        )
        input_data_file.save(input_file_path)
        input_file_paths[param_name] = str(Path(input_file_path))

    return RequestEntry(
        deployment_id,
        module_name,
        function_name,
        request.method,
        request.args,
        input_file_paths,
        datetime.now()
    )

@bp.route('/<deployment_id>/modules/<module_name>/<function_name>', methods=["GET", "POST"])
@bp.route('/<deployment_id>/modules/<module_name>/<function_name>/<filename>', methods=["GET"])
def run_module_function(deployment_id, module_name, function_name, filename=None):
    '''
    Execute the function in WebAssembly module and act based on instructions
    attached to the deployment of this call/execution.
    '''

    if SIM_MAX_LATENCY_SECS is not None:
        waitsecs = random.random() * SIM_MAX_LATENCY_SECS
        logger.info('%f second simulated latency on execution route', waitsecs)
        time.sleep(waitsecs)

    if filename:
        # If a filename is passed, this route works merely for file serving.
        # NOTE: Intented to use by __modules__, not e.g. end users.
        return send_file(module_mount_path(module_name, filename))

    entry = prepare_request_entry(deployment_id, module_name, function_name)

    if name_is_local(deployment_id, module_name, function_name):
        # Assume that the work wont take long and do it synchronously on GET.
        if request.method.lower() == 'get':
            exec_log(entry)
        else:
            # Send data to worker thread to handle non-blockingly.
            wasm_queue.put(entry)

        # Return a link to this request's result (which could link further until
        # some useful value is found).
        return jsonify({ 'resultUrl': results_route(entry.request_id, full=True) })

    # At this point, the function must be on another device, so redirect caller
    # there.

    # XXX HACK-y:
    # In order to not get deadlocked on redirecting between peers (i.e., both
    # devices think they are not responsible), respond with 404 if the request
    # is from current host to itself, but was not found locally (because not yet
    # deployed).
    rargs = dict(request.args)
    if "__wasmiot_rpc" in rargs:
        rargs.pop("__wasmiot_rpc")
        if rargs["__wasmiot_rpc"] == "local":
            return endpoint_failed(request, "risk of endless peer-redirections; maybe in the middle of redeployment", 404)

    # NOTE: Just calling redirect(url) would be a bit nicer (client
    # could connect to the actual server directly), but web browser
    # prevents making calls to different hosts with Javascript. Calling
    # the endpoint here at the server goes around the browser, but still
    # we TODO should try to deal securely with same-origin policy...
    logger.debug('Redirecting execution of %r/%r to a peer', module_name, function_name)
    primitive, files_bytes = deployments[deployment_id].m2m(
        module_name,
        function_name,
        request.data,
        list(map(int, rargs.values())),
    )
    # Write the results to local and respond like it originated
    # from this server in order to TODO go around browser same-origin policy.
    temp_name = "temp_name"
    redir_result_file = f"{temp_name}._redirected"
    # Use the directory of the entry.
    redirect_result_path = per_request_file_path(entry.request_id, redir_result_file)
    redirect_result_path.parent.mkdir(exist_ok=True, parents=True)
    with open(redirect_result_path, 'wb') as f:
        f.write(files_bytes)

    entry.redirected = True
    entry.success = True
    entry.result = [
        primitive,
        [
            results_route(
                entry.request_id,
                full=True,
                file=redir_result_file,
            )
        ]
    ]

    request_history.append(entry)
    return jsonify({
        "resultUrl": results_route(entry.request_id, full=True)
    })


@bp.route('/<deployment_id>/stream/modules/<module_name>/<function_name>', methods=["GET"])
def stream_module_function(deployment_id, module_name, function_name):
    '''
    Execute the function in WebAssembly module but __immediately__ return the
    results as a continuous byte stream instead of returning a link to the
    eventual result location.

    TODO: Actually send a "stream" meaning a single user request is needed to
    initiate continouous calls of the same WebAssembly work.
    '''
    entry = prepare_request_entry(deployment_id, module_name, function_name)

    if name_is_local(deployment_id, module_name, function_name):
        # Assume that the work wont take long and do it synchronously on GET.
        exec_log(entry)
        if not entry.success:
            return endpoint_failed(request, "failure running Wasm:" + entry.result)
        primitive, files = entry.result

        # Concat all the outputs into a single byte stream with primitives first
        # and files second (in whatever order they come out in).
        data = bytearray([primitive])

        for mount in files:
            file = per_request_file_path(entry.request_id, mount.path)
            with open(file, 'rb') as f:
                content = f.read()
                data += content

        return Response(bytes(data), mimetype='application/octet-stream')

    return endpoint_failed(request, 'resource does not exist', 404)


@bp.route('/<deployment_id>/migrate/<module_name>', methods=['POST'])
def evict_module(deployment_id, module_name):
    '''
    Request starting the process of moving a module away from the device it
    currently resides on.
    '''
    if name_is_local(deployment_id, module_name):
        logger.info('Requesting the eviction of module "%s" held locally', module_name)
        device_to_evict_from = FLASK_APP
    elif peer_has_module(deployment_id, module_name):
        logger.info('Requesting the eviction of module "%s" from the peer device currently holding it', module_name)
        # Let orchestrator find where the module currently is.
        device_to_evict_from = None
    else:
        return endpoint_failed(request, 'module not found', 404)

    deployment = deployments[deployment_id]
    # Send migration-request to the deployment's original creator.
    migration_url = deployment.orchestrator_address + f'migrate/{deployment_id}/{module_name}'
    res = requests.post(
        migration_url,
        data={'from': device_to_evict_from},
        timeout=5,
    )
    if res.ok:
        return jsonify({'status': 'success'})

    return endpoint_failed(
        request, f'migration failed on orchestrator at "{migration_url}": {str(res)}')


@bp.route('/deploy/<deployment_id>', methods=['DELETE'])
def deployment_delete(deployment_id):
    '''
    Forget the given deployment.
    '''
    if deployment_id in deployments:
        del deployments[deployment_id]
        return jsonify({'status': 'success'})
    return endpoint_failed(request, 'deployment does not exist', 404)


def updated_configs(deployment, incoming_modules: list[ModuleConfig]) -> list[ModuleConfig]:
    '''
    For a deployment's modules:
    - Add new ones.
    - Keep old ones _intact_ if they still are listed in new ones.
    - Remove old ones no longer needed.

    Return updated list of module configs.
    '''
    deployed_module_names = {m.name for m in incoming_modules}
    # NOTE: To handle migrations, keep the already existing module configurations intact.
    existing_module_names = {m.name for m in deployment.modules.values()}
    new_module_configs = filter(
        lambda x: x.name not in existing_module_names,
        incoming_modules
    )
    # Remove those old modules, that are _not_ in the new deployment.
    old_module_configs = filter(
        lambda x: x.name in deployed_module_names,
        deployment._modules
    )
    return list(new_module_configs) + list(old_module_configs)


def new_deployment(manifest, old_one: Deployment | None = None) -> Deployment:
    '''
    Create a new deployment based on manifest. Reuse some existing resources
    from an old deployment.
    '''
    modules = manifest['modules']

    if not modules:
        return jsonify({'message': 'No modules listed'})

    try:
        module_configs = fetch_modules(modules)
    except FetchFailures as err:
        logger.error("Failed fetching modules", exc_info=True)
        return endpoint_failed(
            request,
            msg=f'{len(err.errors)} fetch failures',
            status_code=500,
            errors=err.errors
        )

    # Initialize __separate__ execution environments for each module for this
    # deployment, adding filepath roots for the modules' directories that they
    # are able to use. This way when file-access is granted via runtime, modules
    # will only access their own directories.

    if old_one:
        module_configs = updated_configs(old_one, module_configs)

    runtimes = {}
    for mod in module_configs:
        # Do not re-initialize existing runtimes.
        if old_one and (mod.name in old_one.runtimes):
            runtimes[mod.name] = old_one.runtimes[mod.name]
        else:
            runtimes[mod.name] = WasmtimeRuntime([str(module_mount_path(mod.name))])

    return Deployment(
        manifest["orchestratorApiBase"],
        manifest["deploymentId"],
        runtimes=runtimes,
        endpoints=manifest["endpoints"],
        peers=manifest["peers"],
        _modules=module_configs,
        _instructions=manifest["instructions"],
        _mounts=manifest["mounts"],
    )

def prepare_deployment() -> str:
    '''Add a new prepared deployment based on request'''
    data = request.get_json(silent=True)
    if not data:
        raise Exception('Non-existent or malformed deployment data')

    deployment_id = data["deploymentId"]
    prepared_deployments[deployment_id] = new_deployment(data, deployments.get(deployment_id, None))
    return deployment_id


@bp.route('/deploy', methods=['POST'])
def deployment_create():
    '''
    Request content-type needs to be 'application/json'
    - POST: Parses the deployment from request and enacts it.
    '''
    try:
        deployment_id = prepare_deployment()
    except Exception as err:
        return endpoint_failed(request, str(err), 400)

    # Remove the prepared deployment immediately, putting it into use.
    deployments[deployment_id] = prepared_deployments.pop(deployment_id)
    deployment_locks[deployment_id] = threading.Lock()

    return jsonify({'status': 'success'})


@bp.route('/deploy/prepare', methods=['POST'])
def deployment_prepare():
    '''
    Create a new deployment but hold it in a temporary collection, making it
    not usable yet by execution requests.
    
    Used to redeploy mid execution.
    '''
    try:
        _ = prepare_deployment()
    except Exception as err:
        return endpoint_failed(request, str(err), 400)

    return jsonify({'status': 'success'})


@bp.route('/deploy/release/<deployment_id>', methods=['PUT'])
def deployment_release(deployment_id):
    '''
    Replace a deployment with a new one that was put on hold previously,
    allowing execution requests to use the updated version.
    '''
    # Prevent any requests to this deployment from starting before new
    # instructions are set.
    if deployment_id in deployment_locks:
        deployment_locks[deployment_id].acquire()

    # Set the new instructions.
    deployments[deployment_id] = prepared_deployments.pop(deployment_id)

    # Allow the requests to continue.
    if deployment_id in deployment_locks:
        deployment_locks[deployment_id].release()

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
        # Map the names of data files to their responses. The names are used to
        # save the files on disk for the module to use.
        res_others = {
            name: requests.get(url, timeout=5)
            for name, url in module.get("urls", {}).get("other", {}).items()
        }

        # Check that each request succeeded before continuing on.
        # Gather errors together.
        errors = []
        for res in itertools.chain([res_bin], res_others.values()):
            if not res.ok:
                errors.append(res)

        if errors:
            raise FetchFailures(errors)

        # "Request for module by name"
        module_path = os.path.join(current_app.config["MODULE_FOLDER"], module["name"])
        # Confirm that the module directory exists and create it if not TODO:
        # This would be better performed at startup.
        os.makedirs(current_app.config["MODULE_FOLDER"], exist_ok=True)
        with open(module_path, 'wb') as filepath:
            filepath.write(res_bin.content)

        # Add other listed files related to the module.
        data_files = {}
        for key, res_other in res_others.items():
            other_path = module_mount_path(module["name"], key)

            other_path.parent.mkdir(exist_ok=True, parents=True)
            with open(other_path, 'wb') as filepath:
                filepath.write(res_other.content)

            # Map the mount name to whatever path the actual file is at.
            data_files[key] = other_path

            # update the module configuration with the model path
            # TODO: Does having a "model path" attribute in the module
            # config have direct benefits over "generic" files list?
            # Something must point out the currently used model file for ML modules:
            # - a) either it is always assumed to be the first data file
            # - b) or there is a "special" attribute for the model
            # - c) alternatively there is some additional information about the data files,
            #   - type or description, that would indicate the model file
            # new_module_config.ml_model = MLModel(other_path)

        # Save downloaded module's details.
        new_module_config = ModuleConfig(
            id=module["id"],
            name=module["name"],
            path=module_path,
            data_files=data_files,
        )
        # combining options a) and b) from above:
        new_module_config.set_model_from_data_files()

        # Create the params directory for this module's files.
        new_module_mount_path = Path(module_mount_path(new_module_config.name))
        if not new_module_mount_path.exists():
            os.makedirs(new_module_mount_path, exist_ok=True)

        configs.append(new_module_config)

    return configs
