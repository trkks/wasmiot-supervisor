# wasmiot-supervisor
## Requirements
- [Python3](https://www.python.org/downloads/)
- Linux (for Windows users [WSL](https://learn.microsoft.com/en-us/windows/wsl/install))

### raspberry pi

For opencv support there are necessary requirements that need to be installed before installing the python libraries:
```
sudo apt install libwayland-cursor0 libxfixes3 libva2 libdav1d4 libavutil56 libxcb-render0 libwavpack1 libvorbis0a libx264-160 libx265-192 libaec0 libxinerama1 libva-x11-2 libpixman-1-0 libwayland-egl1 libzvbi0 libxkbcommon0 libnorm1 libatk-bridge2.0-0 libmp3lame0 libxcb-shm0 libspeex1 libwebpmux3 libatlas3-base libpangoft2-1.0-0 libogg0 libgraphite2-3 libsoxr0 libatspi2.0-0 libdatrie1 libswscale5 librabbitmq4 libhdf5-103-1 libharfbuzz0b libbluray2 libwayland-client0 libaom0 ocl-icd-libopencl1 libsrt1.4-gnutls libopus0 libxvidcore4 libzmq5 libgsm1 libsodium23 libxcursor1 libvpx6 libavformat58 libswresample3 libgdk-pixbuf-2.0-0 libilmbase25 libssh-gcrypt-4 libopenexr25 libxdamage1 libsnappy1v5 libsz2 libdrm2 libxcomposite1 libgtk-3-0 libepoxy0 libgfortran5 libvorbisenc2 libopenmpt0 libvdpau1 libchromaprint1 libpgm-5.3-0 libcairo-gobject2 libavcodec58 libxrender1 libgme0 libpango-1.0-0 libtwolame0 libcairo2 libatk1.0-0 libxrandr2 librsvg2-2 libopenjp2-7 libpangocairo-1.0-0 libshine3 libxi6 libvorbisfile3 libcodec2-0.9 libmpg123-0 libthai0 libudfread0 libva-drm2 libtheora0
```

The requirements were taken from here: https://www.piwheels.org/project/opencv-contrib-python/ (python 3.9 and armv7l for raspberry pi 4)

## Installation

Clone the project:
```
git clone git@github.com:LiquidAI-project/wasmiot-supervisor.git
```

Install requirements. You might want to install them in a [virtual environment](https://docs.python.org/3/library/venv.html).

```
# Create
python3 -m venv my-supervisor-venv
# Activate
source my-supervisor-venv/bin/activate
```

Finally installing is done with:
```
pip install -r requirements.txt
```

Set up [Sentry](https://sentry.io) logging (optional):
```
export SENTRY_DSN="<your sentry-dsn here>"
```

Run with:
```
cd host_app
python __main__.py
```

Now the supervisor should be accessible at [`http://localhost:5000/`](http://localhost:5000/).
```
curl http://localhost:5000/
```
The supervisor's logs in your terminal should show that a `GET` request was received.


## Testing deployment

For testing the supervisor you need to provide it with a "deployment manifest" and the WebAssembly modules to run along with their descriptions.
The modules can be found in the [wasmiot-modules repo](https://github.com/LiquidAI-project/wasmiot-modules) (see the link for build instructions).

The simplest deployment to test is counting the Fibonacci sequence with the `fibo` module.

### Command line

After building the WebAssembly modules, start up a simple file server inside the `modules` directory:
```
cd modules
python3 -m http.server
```
This will allow the supervisor to fetch needed files on deployment.

Using `curl` you can deploy the `fibo` module with the following command containing the needed manifest as data:
```bash
curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{
        "deploymentId":"0",
        "modules":[
            {
                "id":"0",
                "name":"fibo",
                "urls":{
                    "binary":"http://localhost:8000/wasm-binaries/wasm32-unknown-unknown/fibo.wasm",
                    "description":"http://localhost:8000/fibo/open-api-description.json",
                    "other":[]
                }
            }
        ],
        "instructions": [
            {
                "sequence": 0,
                "to": null
            }
        ]
    }' \
    http://localhost:5000/deploy
```

Then on success, you can count the (four-byte representation of) 7th Fibonacci number with the command:
```bash
curl localhost:5000/0/modules/fibo/fibo?iterations=7
```

### Web GUI

You need to upload to fibo module to [the orchestrator](https://github.com/LiquidAI-project/wasmiot-orchestrator) (see the link for installation), let's say with the name 'Fibo'. And
make a deployment named for example 'fibo-dep' with call sequence
'Fibo:fibo'.

First you need to find out which device the package was sent to. This can
be done by checking either the orchestrator logs or from the orchestrator
web interface from the deployments page.

If you are running the docker test environment, meaning the devices are
just docker containers, you will then need to check which port the device
is mapped to. You can do this by running
```
docker container list
```

Afterwards you can run the module via the supervisor REST-interface by
issuing GET-request to the url
```
'\<host\>:\<port\>/modules/\<module name\>/\<function name\>?param1=\<integer number\>
```
which in the docker test environment case
is 'localhost:\<port\>/modules/Fibo/fibo?param1=\<integer number\>'.
You can just do this by accessing the URL with your browser and you should get json containing the resulting number as a response.

## Testing ML deployment

You first need to deploy a wasm-module to the device from the orchestrator. As a test module you can use the example from [here](https://github.com/radu-matei/wasi-tensorflow-inference).
That repository contains the code for the wasm-module (source in crates/wasi-mobilenet-inference pre-compiled binary in model/optimized-wasi.wasm) and the model file
(in model/mobilenet_v2_1.4_224_frozen.pb).

After deploying the module to the device, you need to supply the model file, which currently can't be done via orchestrator. Instead you can use curl, eg.:
```
curl -v -X POST -F model=@./model/mobilenet_v2_1.4_224_frozen.pb http://localhost:5000/ml/model/mobilenet
```
After which you can test the inference via curl, eg.:
```
curl -v -X POST -F data=@./testdata/husky.jpeg http://localhost:5000/ml/mobilenet
```
You can find a couple of test images in the wasi-inference repository in 'testdata'

## Devcontainer
Use VSCode for starting in container. NOTE: Be sure the network it uses is
created i.e., before starting the container run:
```
docker network create wasmiot-net
```

To build the devcontainer image manually, run:
```
docker build -t ghcr.io/liquidai-project/wasmiot-supervisor:devcontainer --target vscode-devcontainer .
```
