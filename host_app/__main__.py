#from .app import create_app, teardown_zeroconf
import os

# Have to setup environment variables before importing flask app
os.environ.setdefault("FLASK_APP", "host_app")
os.environ.setdefault("FLASK_ENV", "development")
os.environ.setdefault("FLASK_DEBUG", "1")

from .utils.configuration import INSTANCE_PATH

from host_app.flask_app import app as flask_app

if __name__ == "__main__":
    print("Starting program")

    debug = bool(os.environ.get("FLASK_DEBUG", 0))

    #print('starting modules')
    #wasm_daemon = threading.Thread(name='wasm_daemon',
    #                               daemon=True,
    #                               target=wa.start_modules,
    #                                 )
    #wasm_daemon.start()

    app = flask_app.create_app(instance_path=INSTANCE_PATH)

    app.run(debug=debug, host="0.0.0.0", use_reloader=False)

