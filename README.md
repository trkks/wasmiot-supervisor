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

Some simple test modules can be found in the
`wasmiot-orchestrator` project's directory `/client/testData/wasm`.
An all-round deployment to test is deploying the `abc` module which has three
functions `a`, `b` and `c` outputting numbers "seeded" by files (or
_mounts_) deployed and sent in HTTP requests.

Trying to use the supervisor directly is not recommended and instead use the
command line interface provided by `wasmiot-orchestrator`.

### Interacting through orchestrator

- Start the orchestrator (see [wasmiot-orchestrator](https://github.com/LiquidAI-project/wasmiot-orchestrator) for instructions).
    - in these instructions the orchestrator is assumed be at `http://localhost:3000`
- Install and start the supervisor (see [installation](./README.md#installation)).
    - in these instructions the supervisor is assumed be at `http://localhost:5000` with a device name `my-device`
- From the orchestrator's API check that `my-device` is discovered by the orhestrator: [http://localhost:3000/file/device](http://localhost:3000/file/device)


#### Using the orchestrator CLI
- The CLI tool is found at the `wasmiot-orchestrator` project under
`/client/cli/` (see contained `README.md` for usage with Docker). There are
`npm` scripts provided for compiling (from TypeScript ) and running this tool.

- __In these instructions the `npm` scripts are assumed to be run inside Docker as per the CLI README.__

- Create a new module with the orchestrator (Module creation):
    - `npm run -- client module create abcmod </path/to/abc.wasm>`
        - Choose `.wasm` compiled with `cargo --target wasm32-wasi`

- Provide description and mount-files for the new module (Module description):
    ```bash
    npm run -- client module desc abcmod </path/to/abc/description.json> -m deployFile -p </what/ever/file>
    ```
    - Note that this step can be performed again multiple times with different
    file argument for `-p` and the change will reflect on the existing module.

- Create a new deployment:
    ```bash
    npm run -- client deployment create abcdep -d -m abcmod -f <a|b|c>
    ```
    - The `-d` option can be left empty for automatic device selection of explicitly specified `-d my-device`.

- Deploy the new module and mount files to the device:
    ```bash
    npm run -- client deployment deploy abcdep
    ```

- Test the `abcdep` deployment with the orchestrator. The arguments needed are
dependent on the function you chose at deployment creation. For example in the
case of `-f a` you also need to pass in a file for the `execFile` mount:
    ```bash
    npm run -- client execute abcdep '{"param0": 42, "param1": 69}' -m execFile -p </what/ever/file>
    ```
    - The response should be JSON containing a link to the result struct stored
    at `my-device`. This result struct will then contain a `result` field with
    your computed number and `success` field indicating if the process was
    successfully completed by the supervisor.

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