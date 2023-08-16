from wasmtime import Config, Store, Engine, Linker, WasiConfig, Module, FuncType, Func, Instance
from wasmtime.ValType import i32, i64, f32, f64
from time import sleep
import cv2

# Create a new config for enabling wasmtime features
cfg = Config()

# Create the engine with the configuration
engine = Engine(cfg)

# Create a linker with the engine. This will house all our imports for Wasm modules
linker = Linker(engine)
linker.define_wasi()

# Create a store with the engine
store = Store(engine)

wasi = WasiConfig()
wasi.inherit_stdout()
wasi.inherit_env()

store.set_wasi(wasi)


# Functions to link to Wasm modules

def wt_python_clock_ms():
    return int(round(time() * 1000))

def wt_python_delay(d):
    sleep(d/1000.0)

def wt_python_print(pointer, length):
    mem = rt.get_memory(0)
    msg = mem[pointer:pointer + length].tobytes().decode()
    print(msg, end="")

def wt_python_println(msg):
    print(msg + "\n")

def wt_python_printInt(n):
    print(n, end="")

def wt_python_takeImage(data_ptr):
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    cam.release()

    mem = rt.get_memory(0)
    data = np.array(img).flatten().tobytes()
    mem[data_ptr:data_ptr + len(data)] = data

def wt_python_rpcCall(func_name_ptr, func_name_size, data_ptr, data_size):
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

def wt_python_getTemperature():
    import adafruit_dht
    import board
    try:
        dhtDevice = adafruit_dht.DHT22(board.D4)
        temperature = dhtDevice.temperature
        return temperature
    except Exception as error:
        print(error.args[0])

def wt_python_getHumidity():
    import adafruit_dht
    import board
    try:
        dhtDevice = adafruit_dht.DHT22(board.D4)
        humidity = dhtDevice.humidity
        return humidity
    except Exception as error:
        print(error.args[0])
    
# Link functions to Wasm modules
def link_functions():
    sys = "sys"
    http = "http"
    communication = "communication"
    dht = "dht"
    camera = "camera"

    # system functions
    linker.define_func(sys, "millis", FuncType([],[i32()]), wt_python_clock_ms)
    linker.define_func(sys, "delay", FuncType([i32()],[]), wt_python_delay)
    linker.define_func(sys, "print", FuncType([i32(),i32()],[]), wt_python_print)
    linker.define_func(sys, "println", FuncType([i32()], []), wt_python_println)
    linker.define_func(sys, "printInt", FuncType([i32()], []), wt_python_printInt)

    # communication
    linker.define_func(communication, "rpcCall", FuncType([i32(), i32(), i32(), i32()], []), wt_python_rpcCall)

    # peripheral
    linker.define_func(camera, "takeImage", FuncType([i32], []), wt_python_takeImage)
    linker.define_func(dht, "getTemperature", FuncType([], [f32()]), wt_python_getTemperature)
    linker.define_func(dht, "getHumidity", FuncType([], [f32()]), wt_python_getHumidity)
