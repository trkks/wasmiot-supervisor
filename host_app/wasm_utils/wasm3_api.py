import wasm3
from time import sleep,time

env = wasm3.Environment()
rt = env.new_runtime(4096)


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

    # system functions
    mod.link_function("sys", "millis", "i()", m3_python_clock_ms)
    mod.link_function("sys", "delay", "v(i)", m3_python_delay)
    mod.link_function("sys", "print", "v(*i)", m3_python_print)
    mod.link_function("sys", "println", "v(*)", m3_python_println)
    mod.link_function("sys", "printInt", "v(i)", m3_python_printInt)

    # peripheral
    mod.link_function("dht", "getTemperature", "f()", m3_python_getTemperature)
    mod.link_function("dht", "getHumidity", "f()", m3_python_getHumidity)

