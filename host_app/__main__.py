#from .app import create_app, teardown_zeroconf
# import wasm_utils.wasm_utils as wa
import os
import threading

# Have to setup environment variables before importing flask app
os.environ.setdefault("FLASK_APP", "thingi")
os.environ.setdefault("FLASK_ENV", "development")
os.environ.setdefault("FLASK_DEBUG", "1")

import flask_app.app as flask_app

if __name__ == "__main__":
    print("Starting program")


    #print('starting modules')
    #wasm_daemon = threading.Thread(name='wasm_daemon',
    #                               daemon=True,
    #                               target=wa.start_modules,
    #                                 )
    #wasm_daemon.start()

    app = flask_app.create_app()

    app.run(debug=True, host="0.0.0.0")

