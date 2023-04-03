# wasmiot-supervisor

Install requirements:
```
pip install -r requirements.txt
```

Run with (requires python3)
```
cd host_app
python __main__.py
```
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