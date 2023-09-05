"""General utilities for Wasm."""

import os
import platform
import struct
from time import sleep, time
from typing import Any, Callable, Optional

import cv2
import requests

from host_app.utils.configuration import remote_functions
from host_app.wasm_utils.wasm_api import WasmRuntime

if platform.system() != "Windows":
    import adafruit_dht
    import board


class RemoteFunction:
    """Superclass for remote function generator."""
    def __init__(self, runtime: WasmRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> WasmRuntime:
        """Get the runtime."""
        return self._runtime

    @property
    def function(self) -> Callable[..., Any]:
        """Get the remote function as a callable."""
        raise NotImplementedError


def python_clock_ms():
    """Return the current epoch time in milliseconds."""
    return int(round(time() * 1000))


def python_delay(delay: int) -> None:
    """Sleep for the specified number of milliseconds."""
    sleep(delay / 1000.0)


def python_println(message: str) -> None:
    """Print the specified message with a newline character."""
    print(message + "\n")


def python_print_int(number: int) -> None:
    """Print the specified integer."""
    print(number, end="")


def python_get_temperature() -> float:
    """Get the temperature from the DHT22 sensor."""
    if platform.system() == "Windows":
        return 0.0

    try:
        dht_device = adafruit_dht.DHT22(board.D4)
        temperature = dht_device.temperature
        if temperature is None:
            return 0.0
        return float(temperature)
    except RuntimeError as error:
        print(error.args[0])
        return 0.0


def python_get_humidity() -> float:
    """Get the humidity from the DHT22 sensor."""
    if platform.system() == "Windows":
        return 0.0

    try:
        dht_device = adafruit_dht.DHT22(board.D4)  # ignore=reportUnboundVariable
        humidity = dht_device.humidity
        if humidity is None:
            return 0.0
        return float(humidity)
    except RuntimeError as error:
        print(error.args[0])
        return 0.0


class Print(RemoteFunction):
    """Remote function generator for printing."""
    @property
    def function(self) -> Callable[[int, int], None]:
        """Print the string decoded from the specified memory location at the given runtime."""
        def python_print(pointer: int, length: int) -> None:
            """Print the string decoded from the specified memory location at the given runtime."""
            data, error = self.runtime.read_from_memory(pointer, length)
            message = data.decode() if error is None else error
            print(message, end="")

        return python_print


class TakeImage(RemoteFunction):
    """Remote function generator for capturing image with attached camera."""

    def alloc(self, nbytes: int):
        """Allocate nbytes of memory in the runtime."""
        return self.runtime.run_function("alloc", [nbytes], self.runtime.current_module_name)

    @property
    def function(self) -> Callable[[int, int], None]:
        def python_take_image(out_ptr_ptr: int, out_size_ptr: int):
            """
            Take an image and write it to memory at the given runtime using the given module.

            Store the pointer to the image in out_ptr_ptr and the size as in
            out_size_ptr both in 32bits LSB.
            """
            cam = cv2.VideoCapture(0)  # type: ignore
            _, img = cam.read()
            cam.release()

            _, datatmp = cv2.imencode(".jpg", img)
            data = datatmp.tobytes()
            data_len = len(data)
            data_ptr = self.alloc(data_len)
            if data_ptr is None:
                # memory allocation failed
                raise MemoryError(f"Unable to allocate {data_len} bytes of memory!")

            # Write the image to memory.
            self.runtime.write_to_memory(data_ptr, data, self.runtime.current_module_name)

            # Write pointers to image pointer and length to memory assuming they both
            # have 32 bits allocated.
            pointer_bytes = struct.pack("<I", data_ptr)
            length_bytes = struct.pack("<I", data_len)
            self.runtime.write_to_memory(out_ptr_ptr, pointer_bytes, self.runtime.current_module_name)
            self.runtime.write_to_memory(out_size_ptr, length_bytes, self.runtime.current_module_name)

        return python_take_image

class RpcCall(RemoteFunction):
    """Remote function generator for RPC calls."""
    @property
    def function(self) -> Callable[[int, int, int, int], None]:
        """Make a POST request with data.
        Both the data and the target host is determined from the runtime memory."""
        def python_rpc_call(func_name_ptr: int, func_name_size: int,
                            data_ptr: int, data_size: int) -> None:
            func_name_bytes, error = self.runtime.read_from_memory(func_name_ptr, func_name_size)
            if error is not None:
                print(error)
                return
            func_name = func_name_bytes.decode()
            print(func_name)
            func = remote_functions[func_name]
            data, error = self.runtime.read_from_memory(data_ptr, data_size)
            if error is not None:
                print(error)
                return
            files = [("img", data)]

            response = requests.post(
                url=func["host"],
                files=files,
                timeout=120
            )
            print(response.text)

        return python_rpc_call


class RandomGet(RemoteFunction):
    """Remote function generator for writing random bytes to runtime memory."""
    @property
    def function(self) -> Callable[[int, int], int]:
        class WasiErrno:
            """WASI errno codes."""
            SUCCESS = 0
            BADF = 8
            INVAL = 28

        def random_get(buf_ptr: int, size: int) -> int:
            """Generate random bytes and write them to the specified memory location."""
            self.runtime.write_to_memory(buf_ptr, os.urandom(size))
            return WasiErrno.SUCCESS

        return random_get
