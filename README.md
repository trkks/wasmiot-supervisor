# wasmiot-supervisor

## Installation

Install requirements:
```
pip install -r requirements.txt
```

Run with (requires python3)
```
cd host_app
python __main__.py
```
### Installing on raspberry pi

For opencv support there are necessary requirements that need to be installed before installing the python libraries:
```
sudo apt install libwayland-cursor0 libxfixes3 libva2 libdav1d4 libavutil56 libxcb-render0 libwavpack1 libvorbis0a libx264-160 libx265-192 libaec0 libxinerama1 libva-x11-2 libpixman-1-0 libwayland-egl1 libzvbi0 libxkbcommon0 libnorm1 libatk-bridge2.0-0 libmp3lame0 libxcb-shm0 libspeex1 libwebpmux3 libatlas3-base libpangoft2-1.0-0 libogg0 libgraphite2-3 libsoxr0 libatspi2.0-0 libdatrie1 libswscale5 librabbitmq4 libhdf5-103-1 libharfbuzz0b libbluray2 libwayland-client0 libaom0 ocl-icd-libopencl1 libsrt1.4-gnutls libopus0 libxvidcore4 libzmq5 libgsm1 libsodium23 libxcursor1 libvpx6 libavformat58 libswresample3 libgdk-pixbuf-2.0-0 libilmbase25 libssh-gcrypt-4 libopenexr25 libxdamage1 libsnappy1v5 libsz2 libdrm2 libxcomposite1 libgtk-3-0 libepoxy0 libgfortran5 libvorbisenc2 libopenmpt0 libvdpau1 libchromaprint1 libpgm-5.3-0 libcairo-gobject2 libavcodec58 libxrender1 libgme0 libpango-1.0-0 libtwolame0 libcairo2 libatk1.0-0 libxrandr2 librsvg2-2 libopenjp2-7 libpangocairo-1.0-0 libshine3 libxi6 libvorbisfile3 libcodec2-0.9 libmpg123-0 libthai0 libudfread0 libva-drm2 libtheora0
```

The requirements were taken from here: https://www.piwheels.org/project/opencv-contrib-python/ (python 3.9 and armv7l for raspberry pi 4)

## Testing deployment

The simplest deployment to test is the 'fibo' module. You need to upload to fibo module to the orchestrator, let's say with the name 'Fibo'. And
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

## Devcontainer
Use VSCode for starting in container. NOTE: Be sure the network it uses is
created i.e., before starting the container run:
```
docker network create wasmiot-net
```
