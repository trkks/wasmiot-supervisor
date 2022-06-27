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

def link_functions(mod):
    sys = "sys"
    http = "http"
    communication = "communication"

    # system functions
    mod.link_function("sys", "millis", "i()", m3_python_clock_ms)
    mod.link_function("sys", "delay", "v(i)", m3_python_delay)
    mod.link_function("sys", "print", "v(*i)", m3_python_print)
    mod.link_function("sys", "println", "v(*)", m3_python_println)
    mod.link_function("sys", "printInt", "v(i)", m3_python_printInt)

