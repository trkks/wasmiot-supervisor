import wasm3
from time import sleep,time
import requests
import cv2
import numpy as np

from host_app.utils.configuration import remote_functions, modules

env = wasm3.Environment()
rt = env.new_runtime(4096)

#remote_functions = {
#    'convert2Grayscale': {
#        'host': 'http://localhost:5000/img/test/convert2Grayscale',
#        'token': None
#    }
#}

def m3_python_clock_ms():
    return int(round(time() * 1000))

def m3_python_delay(d):
    sleep(d/1000.0)

def m3_python_print(pointer, length):
    mem = rt.get_memory(0)
    msg = mem[pointer:pointer + length].tobytes().decode()
    print(msg, end="")

def m3_python_println(msg):
    print(msg + "\n")

def m3_python_printInt(n):
    print(n, end="")

def m3_python_takeImage(data_ptr):
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    cam.release()

    mem = rt.get_memory(0)
    data = np.array(img).flatten().tobytes()
    mem[data_ptr:data_ptr + len(data)] = data

def m3_python_rpcCall(func_name_ptr, func_name_size, data_ptr, data_size):
    mem = rt.get_memory(0)
    print(func_name_ptr)
    print(func_name_size)
    print(mem[func_name_ptr:func_name_ptr + func_name_size].tobytes().decode())
    func_name = mem[func_name_ptr:func_name_ptr + func_name_size].tobytes().decode()
    func = remote_functions[func_name]
    files = [("img", mem[data_ptr:data_ptr + data_size])]

    response = requests.post(
        func["host"],
        files = files
    )
    print(response.text)

def m3_python_getTemperature():
    import adafruit_dht
    import board
    try:
        dhtDevice = adafruit_dht.DHT22(board.D4)
        temperature = dhtDevice.temperature
        return temperature
    except Exception as error:
        print(error.args[0])

def m3_python_getHumidity():
    import adafruit_dht
    import board
    try:
        dhtDevice = adafruit_dht.DHT22(board.D4)
        humidity = dhtDevice.humidity
        return humidity
    except Exception as error:
        print(error.args[0])

def link_functions(mod):
    sys = "sys"
    http = "http"
    communication = "communication"
    dht = "dht"
    camera = "camera"

    # system functions
    mod.link_function(sys, "millis", "i()", m3_python_clock_ms)
    mod.link_function(sys, "delay", "v(i)", m3_python_delay)
    mod.link_function(sys, "print", "v(*i)", m3_python_print)
    mod.link_function(sys, "println", "v(*)", m3_python_println)
    mod.link_function(sys, "printInt", "v(i)", m3_python_printInt)

    # communication
    mod.link_function(communication, "rpcCall", "v(*iii)", m3_python_rpcCall)

    # peripheral
    mod.link_function(camera, "takeImage", "v(i)", m3_python_takeImage)
    mod.link_function(dht, "getTemperature", "f()", m3_python_getTemperature)
    mod.link_function(dht, "getHumidity", "f()", m3_python_getHumidity)
