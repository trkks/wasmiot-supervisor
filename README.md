# wasmiot-supervisor
## Requirements
- [Python3](https://www.python.org/downloads/)
- Linux (for Windows users [WSL](https://learn.microsoft.com/en-us/windows/wsl/install))
  - `apt install gcc` for installing `pywasm3`

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
python3 -m venv venv
# Activate
source venv/bin/activate
```

Note: if running the supervisor in a Raspberry Pi, uncomment the marked line in `requirements.txt` before running the following command.

Finally installing is done with:
```
pip install -r requirements.txt
```

Set up device configuration files, `device-description.json` and `wasmiot-device-description.json` to `configs` folder. You can use the template configs from the `tempconfigs` folder as a starting point:

```bash
mkdir -p configs
cp -r tempconfigs/* configs
# edit the copied files if necessary
```

Set up [Sentry](https://sentry.io) logging (optional):
```
export SENTRY_DSN="<your sentry-dsn here>"
```

Set up the device name (optional):

```bash
export FLASK_APP=my-device
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

After building the WebAssembly modules, you can start up a simple file server inside the `modules` directory containing `.wasm` files in `wasm-binaries/` when the `build.sh` script is used:
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
## Testing ML deployment

As a test module you can use the example from [here](https://github.com/radu-matei/wasi-tensorflow-inference).
That repository contains the code for the wasm-module (source in crates/wasi-mobilenet-inference pre-compiled binary in model/optimized-wasi.wasm) and the model file
(in model/mobilenet_v2_1.4_224_frozen.pb).

You need to provide both of these files for the supervisor to fetch like with the `fibo` module.

### Without orchestrator
Add the ML-module's files to where your Python HTTP-server (started in the `fibo` test) can serve them from, e.g.:
```
cd modules
curl -L https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/optimized-wasi.wasm > wasm-binaries/wasm32-wasi/ml.wasm
curl -L https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/mobilenet_v2_1.4_224_frozen.pb > wasm-binaries/wasm32-wasi/ml.pb
```

Now deploy the files with:
```bash
curl --header "Content-Type: application/json" --request POST --data '{
        "deploymentId":"1",
        "modules":[
            {
                "id":"1",
                "name":"ml",
                "urls":{
                    "binary":"http://localhost:8000/wasm-binaries/wasm32-wasi/ml.wasm",
                    "description":"http://localhost:8000/object-inference-open-api-description.json",
                    "other":[
                        "http://localhost:8000/wasm-binaries/wasm32-wasi/ml.pb"
                    ]
                }
            }
        ],
        "instructions": [
            {
                "sequence": 0,
                "to": null
            }
        ]
    }' http://localhost:5000/deploy
```

After which you can test the inference with some image file via curl, eg.:
```
# Download a test image
curl -L https://raw.githubusercontent.com/radu-matei/wasi-tensorflow-inference/master/testdata/husky.jpeg > husky.jpeg
# Send the image to supervisor and wait for it to return the classification
curl -v -X POST -F data=@./husky.jpeg localhost:5000/ml/ml
```

The supervisor will again, aften some time, respond with a four-byte representation of a 32-bit integer, that will match the line number in [the labels file](https://github.com/radu-matei/wasi-tensorflow-inference/blob/master/model/labels.txt).

You can find another test image in the [wasi-inference repository in 'testdata'](https://github.com/radu-matei/wasi-tensorflow-inference/tree/master/testdata).

### With orchestrator

- Start the orchestrator (see [wasmiot-orchestrator](https://github.com/LiquidAI-project/wasmiot-orchestrator) for instructions).
    - in these instructions the orchestrator is assumed be at `http://localhost:3000`
- Install and start the supervisor (see [installation](./README.md#installation)).
    - in these instructions the supervisor is assumed be at `http://localhost:5000` with a device name `my-device`
- From the orchestrator's website check that `my-device` is discovered by the orhestrator: [http://localhost:3000/file/device](http://localhost:3000/file/device)
- Create a new module with the orchestrator (Module creation):
    - Name: `fibonacci`
    - Openapi description: raw JSON content from [fibo/open-api-description.json](https://github.com/LiquidAI-project/wasmiot-modules/blob/main/modules/fibo/open-api-description.json)
    - Select `Convert to JSON` and `Submit`
- Push the new module to the orchestrator (Module upload):
    - Select the module: choose `fibonacci`
    - File to upload: choose a compiled `fibo.wasm` file (see [wasmiot-modules](https://github.com/LiquidAI-project/wasmiot-modules) for compilation instructions)
    - Note that you might have to refresh the web page before the `fibonacci` module can be chosen
    - Select `Submit`
- Create a new deployment manifest (Deployment manifest creation):
    - Name: `fibo-dep`
    - Procedure-call sequence: select "Use my-device for fibonacci:fibo" (have only the 1 item in the sequence)
    - Select `Convert to JSON` and `Submit`
- Deploy the new module to the device (Deployment of deployment manifests):
    - Select the deployment manifest: choose `fibo-dep`
    - Note that you might have to refresh the web page before the `fibo-dep` manifest can be chosen
    - Select `Deploy!`
- Test the fibonacci deployment with the orchestrator (Execution):
    - Select the deployment: choose `fibo-dep`
    - Iteration count for fibonacci sequence: 12
    - Select `Execute!`
    - The response should be 233
- Test the fibonacci deployment from the command line:
    - List the deployments using the orchestrator: [http://localhost:3000/file/manifest](http://localhost:3000/file/manifest)
    - Find the item with the name `fibo-dep`
    - From `fullManifest` -> `deploymentId` you should see the deployment id
    - From `fullManifest` -> `endpoints` -> `servers` -> `url` you should see the device address
    - From `fullManifest` -> `endpoints` -> `paths` you should see the path for the fibonacci function
    - From the commandline (replace DEPLOYMENT_ID with the one in your listing):

        ```bash
        curl http://localhost:5000/DEPLOYMENT_ID/modules/fibonacci/fibo?param1=12
        ```

        The answer should be: [233, 0, 0, 0]

- For testing ML inference, create a new module with the orchestrator (Module creation):
    - Name: `mobilenet`
    - Openapi description: raw JSON content from [object-inference-open-api-description.json](https://github.com/LiquidAI-project/wasmiot-modules/blob/main/modules/object-inference-open-api-description.json)
    - Select `Convert to JSON` and `Submit`
- Push the new module to the orchestrator (Module upload):
    - Select the module: choose `mobilenet`
    - File to upload: choose `optimized-wasi.wasm` file (download link: [optimized-wasi.wasm](https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/optimized-wasi.wasm))
    - Select `Submit`
- Create a new deployment manifest (Deployment manifest creation):
    - Name: `mobilenet-dep`
    - Procedure-call sequence: select "Use my-device for mobilenet:alloc" (have only the 1 item in the sequence)
    - Select `Convert to JSON` and `Submit`
- Deploy the module to the device (Deployment of deployment manifests):
    - Select the deployment manifest: choose `mobilenet-dep`
    - Select `Deploy!`
- Upload the ML model to the device using the command line:

    ```bash
    # first download the model from GitHub and then upload it to the device
    curl -L https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/mobilenet_v2_1.4_224_frozen.pb > mobilenet_v2_1.4_224_frozen.pb
    curl -F model=@./mobilenet_v2_1.4_224_frozen.pb http://localhost:5000/ml/model/mobilenet
    ```

- Test the fibonacci deployment from the command line:

    ```bash
    # First download the test image from GitHub and run the inference
    curl -L https://raw.githubusercontent.com/radu-matei/wasi-tensorflow-inference/master/testdata/husky.jpeg > husky.jpeg
    curl -F data=@./husky.jpeg http://localhost:5000/ml/mobilenet
    ```

    The inference result should be 250.

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
