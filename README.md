# wasmiot-supervisor

Wasmiot supervisor is prototype implementation of self-adaptive supervisor for IoT devices. It is based on [WasmTime](https://wasmtime.dev/) and [Flask](https://flask.palletsprojects.com/).

## Installation

Supervisor can be installed either manually on device, or using Docker.

### Docker

Currently there is three (3) variations of images available:
- **ghcr.io/liquidai-project/wasmiot-supervisor:latest** - Latest "stable" version of supervisor. This is the one that should have working feature-set. Version number is the same as the version of the supervisor, and follows [semantic versioning](https://semver.org/) (e.g. `v0.1.0`). Note that at this point no guarantees of backwards compatibility are made.
- **ghcr.io/liquidai-project/wasmiot-supervisor:main** - Latest version of supervisor from `main` branch. This is the one that should be used for testing new features.
- **ghcr.io/liquidai-project/wasmiot-supervisor:devcontainer** - Version of supervisor from `main` branch with VSCode devcontainer support. This is the default image used by VSCode devcontainer.

When running the supervisor in Docker, the automatic discovery of devices is not supported by default docker network. To enable automatic discovery, you can use mdns reflector or create a [macvlan](https://docs.docker.com/network/macvlan/) network.

### Manual installation

#### Requirements
- [Python3](https://www.python.org/downloads/)
- Linux (for Windows users [WSL](https://learn.microsoft.com/en-us/windows/wsl/install))
  - `apt install gcc python3-dev` for installing `pywasm3`

##### raspberry pi

For opencv support there are necessary requirements that need to be installed before installing the python libraries:
```
sudo apt install libwayland-cursor0 libxfixes3 libva2 libdav1d4 libavutil56 libxcb-render0 libwavpack1 libvorbis0a libx264-160 libx265-192 libaec0 libxinerama1 libva-x11-2 libpixman-1-0 libwayland-egl1 libzvbi0 libxkbcommon0 libnorm1 libatk-bridge2.0-0 libmp3lame0 libxcb-shm0 libspeex1 libwebpmux3 libatlas3-base libpangoft2-1.0-0 libogg0 libgraphite2-3 libsoxr0 libatspi2.0-0 libdatrie1 libswscale5 librabbitmq4 libhdf5-103-1 libharfbuzz0b libbluray2 libwayland-client0 libaom0 ocl-icd-libopencl1 libsrt1.4-gnutls libopus0 libxvidcore4 libzmq5 libgsm1 libsodium23 libxcursor1 libvpx6 libavformat58 libswresample3 libgdk-pixbuf-2.0-0 libilmbase25 libssh-gcrypt-4 libopenexr25 libxdamage1 libsnappy1v5 libsz2 libdrm2 libxcomposite1 libgtk-3-0 libepoxy0 libgfortran5 libvorbisenc2 libopenmpt0 libvdpau1 libchromaprint1 libpgm-5.3-0 libcairo-gobject2 libavcodec58 libxrender1 libgme0 libpango-1.0-0 libtwolame0 libcairo2 libatk1.0-0 libxrandr2 librsvg2-2 libopenjp2-7 libpangocairo-1.0-0 libshine3 libxi6 libvorbisfile3 libcodec2-0.9 libmpg123-0 libthai0 libudfread0 libva-drm2 libtheora0
```

The requirements were taken from here: https://www.piwheels.org/project/opencv-contrib-python/ (python 3.9 and armv7l for raspberry pi 4)

## Developing

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

Set up device configuration files, `device-description.json` and `wasmiot-device-description.json` to `./instance/configs` folder. You can use the template configs from the `tempconfigs` folder as a starting point:

```bash
mkdir -p instance/configs
cp -r tempconfigs/* instance/configs
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
python -m host_app
```

Now the supervisor should be accessible at [`http://localhost:5000/`](http://localhost:5000/).
```
curl http://localhost:5000/
```
The supervisor's logs in your terminal should show that a `GET` request was received.

### Versioning

The supervisor uses [semantic versioning](https://semver.org/). The version number is defined in `host_app/_version.py` and `pyproject.toml`. Do not change the version number manually, but use the following command to bump the version number:

```bash
bump-my-version bump [major|minor|patch]
git push origin v$(bump-my-version show current_version)
```

This will update the version number in the files and create a git commit and tag for the new version.

### Devcontainer

Use VSCode for starting in container. NOTE: Be sure the network it uses is
created i.e., before starting the container run:
```
docker network create wasmiot-net
```
NOTE that if you intend to run the devcontainer (or otherwise the supervisor in Docker) alongside orchestrator,
the `wasmiot-net` network should be created by `docker compose` command using __orchestrator's setup__.
So if this is your case, do not run the above command to create the network, but install orchestrator first!

---

To build the devcontainer image manually, run:
```
docker build -t ghcr.io/liquidai-project/wasmiot-supervisor:devcontainer --target vscode-devcontainer .
```

## Testing deployment

For testing the supervisor you need to provide it with a "deployment manifest" and the WebAssembly modules to run along with their descriptions.
The modules can be found in the [wasmiot-modules repo](https://github.com/LiquidAI-project/wasmiot-modules) (see the link for build instructions).
The simplest deployment to test is counting the Fibonacci sequence with the `fibo` module.

After building the WebAssembly modules, you can start up a simple file server inside the `modules` directory containing `.wasm` files in `wasm-binaries/` when the `build.sh` script is used:
### Locally
```
# Define hostname of the server for making requests later.
export WASM_SERVER_HOST=localhost
cd modules
python3 -m http.server
```
### Within Docker network
```
# Define hostname of the server for making requests later.
export WASM_SERVER_HOST=wasm-server
cd modules
# Use the provided script and Dockerfile
./docker-server/run.sh ./docker-server/Dockerfile
```

This will allow the supervisor to fetch needed files on deployment.

Using `curl` you can deploy the `fibo` module with the following command containing the needed manifest as data:
```bash
curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data "{
        \"deploymentId\":\"0\",
        \"modules\":[
            {
                \"id\":\"0\",
                \"name\":\"fiboMod\",
                \"urls\":{
                    \"binary\":\"http://${WASM_SERVER_HOST}:8000/wasm-binaries/wasm32-unknown-unknown/fibo.wasm\",
                    \"description\":\"http://${WASM_SERVER_HOST}:8000/fibo/open-api-description.json\",
                    \"other\":[]
                }
            }
        ],
        \"instructions\": {
            \"modules\": {
                \"fiboMod\": {
                    \"fibo\": {
                        \"paths\": {
                            \"\": {
                                \"get\": {
                                    \"parameters\": [{
                                        \"name\": \"iterations\",
                                        \"in\": \"query\",
                                        \"required\": true,
                                        \"schema\": {
                                            \"type\": \"integer\",
                                            \"format\": \"int64\"
                                        }
                                    }],
                                    \"responses\": {
                                        \"200\": {
                                            \"content\": {
                                                \"application/json\": {
                                                    \"schema\": {
                                                        \"type\": \"integer\",
                                                        \"format\": \"int64\"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        \"to\": null
                    }
                }
            }
        }
    }" \
    http://localhost:5000/deploy
```

Then on success, you can count the (four-byte representation of) 7th Fibonacci number with the command:
```bash
curl localhost:5000/0/modules/fiboMod/fibo?iterations=7
```
The actual value can be found by following the returned URL pointing to the
final computation result.
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
curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data "{
        \"deploymentId\": \"1\",
        \"modules\": [
            {
                \"id\": \"1\",
                \"name\": \"mobilenet\",
                \"urls\": {
                    \"binary\": \"http://${WASM_SERVER_HOST}:8000/wasm-binaries/wasm32-wasi/ml.wasm\",
                    \"description\": \"http://${WASM_SERVER_HOST}:8000/object-inference-open-api-description.json\",
                    \"other\": [
                        \"http://${WASM_SERVER_HOST}:8000/wasm-binaries/wasm32-wasi/ml.pb\"
                    ]
                }
            }
        ],
        \"instructions\": {
            \"modules\": {
                \"mobilenet\": {
                    \"infer_from_ptrs\": {
                        \"paths\": {
                            \"\": {
                                \"get\": {
                                    \"parameters\": [{
                                        \"name\": \"data\",
                                        \"in\": \"requestBody\",
                                        \"required\": true,
                                        \"schema\": {}
                                    }],
                                    \"responses\": {
                                        \"200\": {
                                            \"content\": {
                                                \"application/json\": {
                                                    \"schema\": {
                                                        \"type\": \"integer\",
                                                        \"format\": \"int64\"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        \"to\": null
                    }
                }
            }
        }
    }" \
    http://localhost:5000/deploy
```

After which you can test the inference with some image file via curl, eg.:
```
# Download a test image
curl -L https://raw.githubusercontent.com/radu-matei/wasi-tensorflow-inference/master/testdata/husky.jpeg > husky.jpeg
# Send the image to supervisor and wait for it to return the classification
curl -F data=@./husky.jpeg http://localhost:5000/1/modules/mobilenet/infer_from_ptrs
```

The supervisor will again, aften some time, respond with a URL, that can be
followed to inspect the desired result integer that will match a line number
in [the labels file](https://github.com/radu-matei/wasi-tensorflow-inference/blob/master/model/labels.txt)
of the detected object in the input image.

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
        curl http://localhost:5000/DEPLOYMENT_ID/modules/fibonacci/fibo?iterations=12
        ```

        The answer should contain a link which directs to where the `result` field is 233.

- For testing ML inference, create a new module with the orchestrator (Module creation):
    - Name: `mobilenet`
    - Openapi description: raw JSON content from [object-inference-open-api-description.json](https://github.com/LiquidAI-project/wasmiot-modules/blob/main/modules/object-inference-open-api-description.json)
    - Select `Convert to JSON` and `Submit`
- Push the new module to the orchestrator (Module upload):
    - Select the module: choose `mobilenet`
    - File to upload: choose `optimized-wasi.wasm` file (download link: [optimized-wasi.wasm](https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/optimized-wasi.wasm))
    - Select `Submit`
- Push the ML model to the orchestrator (Module upload):
    - Select the module: choose `mobilenet`
    - File to upload: choose `mobilenet_v2_1.4_224_frozen.pb` file (download link: [mobilenet_v2_1.4_224_frozen.pb](https://github.com/radu-matei/wasi-tensorflow-inference/raw/master/model/mobilenet_v2_1.4_224_frozen.pb))
    - Select `Submit`
- Create a new deployment manifest (Deployment manifest creation):
    - Name: `mobilenet-dep`
    - Procedure-call sequence: select "Use my-device for mobilenet:infer_from_ptrs" (have only the 1 item in the sequence)
    - Select `Convert to JSON` and `Submit`
- Deploy the module to the device (Deployment of deployment manifests):
    - Select the deployment manifest: choose `mobilenet-dep`
    - Select `Deploy!`
- Test the ML deployment (Execution):
    - Select the deployment: choose `mobilenet-dep`
    - From the appearing file-input, upload `husky.jpg` (download link: [husky.jpg](https://raw.githubusercontent.com/radu-matei/wasi-tensorflow-inference/master/testdata/husky.jpeg))
    - Select `Execute!`

    The inference result should be 250.

- The ML deployment can also be tested from the command line:
    ```bash
    # First download the test image from GitHub and run the inference
    curl -L https://raw.githubusercontent.com/radu-matei/wasi-tensorflow-inference/master/testdata/husky.jpeg > husky.jpeg
    curl -F data=@./husky.jpeg http://localhost:5000/DEPLOYMENT_ID/modules/mobilenet/infer_from_ptrs
    ```

## Testing camera module

```bash
curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{
        "deploymentId": "2",
        "modules": [
            {
                "id": "2",
                "name":"camera",
                "urls":{
                    "binary":"http://localhost:8000/wasm-binaries/wasm32-unknown-unknown/camera.wasm",
                    "description":"http://localhost:8000/camera/open-api-description.json",
                    "other":[]
                }
            }
        ],
        "instructions": {
            "modules": {
                "camera": {
                    "take_image": {
                        "paths": {
                            "": {
                                "get": {
                                    "parameters": [],
                                    "responses": {
                                        "200": {
                                            "description": "Return the image taken with a camera from the request",
                                            "content": {
                                                "image/jpeg": {}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "to": null
                    }
                }
            }
        }
    }' \
    http://localhost:5000/deploy
```

To take an image with the camera. The image is save to file `temp_image.jpg`.

```bash
curl http://localhost:5000/2/modules/camera/take_image
```

To see all the results:

```bash
curl http://localhost:5000/request-history
```

## Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@inproceedings{kotilainenProposingIsomorphicMicroservices2022,
  title = {Proposing {{Isomorphic Microservices Based Architecture}} for {{Heterogeneous IoT Environments}}},
  booktitle = {Product-{{Focused Software Process Improvement}}},
  author = {Kotilainen, Pyry and Autto, Teemu and J{\"a}rvinen, Viljami and Das, Teerath and Tarkkanen, Juho},
  editor = {Taibi, Davide and Kuhrmann, Marco and Mikkonen, Tommi and Kl{\"u}nder, Jil and Abrahamsson, Pekka},
  year = {2022},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {621--627},
  publisher = {{Springer International Publishing}},
  address = {{Cham}},
  doi = {10.1007/978-3-031-21388-5_47},
  abstract = {Recent advancements in IoT and web technologies have highlighted the significance of isomorphic software architecture development, which enables easier deployment of microservices in IoT-based systems. The key advantage of such systems is that the runtime or dynamic code migration between the components across the whole system becomes more flexible, increasing compatibility and improving resource allocation in networks. Despite the apparent advantages of such an approach, there are multiple issues and challenges to overcome before a truly valid solution can be built. In this idea paper, we propose an architecture for isomorphic microservice deployment on heterogeneous hardware assets, inspired by previous ideas introduced as liquid software [12]. The architecture consists of an orchestration server and a package manager, and various devices leveraging WebAssembly outside the browser to achieve a uniform computing environment. Our proposed architecture aligns with the long-term vision that, in the future, software deployment on heterogeneous devices can be simplified using WebAssembly.},
  isbn = {978-3-031-21388-5},
  langid = {english},
}
```