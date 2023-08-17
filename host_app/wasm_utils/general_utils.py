"""General utilities for Wasm."""

import os
from time import sleep, time
from typing import Any, Callable

import adafruit_dht
import board
import cv2
import numpy as np
import requests

from utils.configuration import remote_functions
from wasm_utils.wasm_api import WasmRuntime


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
    try:
        dht_device = adafruit_dht.DHT22(board.D4)
        temperature = dht_device.temperature
        return float(temperature)
    except RuntimeError as error:
        print(error.args[0])
        return 0.0


def python_get_humidity() -> float:
    """Get the humidity from the DHT22 sensor."""
    try:
        dht_device = adafruit_dht.DHT22(board.D4)
        humidity = dht_device.humidity
        return float(humidity)
    except RuntimeError as error:
        print(error.args[0])
        return 0.0


class Print(RemoteFunction):
    """Remote function generator for printing."""
    @property
    def function(self) -> Callable[[int, int], None]:
        """Print the string decoded from the specified memory location at the given runtime."""
        def python_print(pointer: int, length: int, runtime: WasmRuntime) -> None:
            """Print the string decoded from the specified memory location at the given runtime."""
            data = runtime.read_from_memory(pointer, length)
            message = data.decode()
            print(message, end="")

        return python_print


class TakeImage(RemoteFunction):
    """Remote function generator for printing."""
    @property
    def function(self) -> Callable[[int], None]:
        def python_take_image(data_ptr: int, runtime: WasmRuntime) -> None:
            """Take an image and write it to the specified memory location at the given runtime."""
            cam = cv2.VideoCapture(0)
            _, img = cam.read()
            cam.release()

            data = np.array(img).flatten().tobytes()
            runtime.write_to_memory(data_ptr, data)

        return python_take_image


class RpcCall(RemoteFunction):
    """Remote function generator for RPC calls."""
    @property
    def function(self) -> Callable[[int, int, int, int], None]:
        """Make a POST request with data.
        Both the data and the target host is determined from the runtime memory."""
        def python_rpc_call(func_name_ptr: int, func_name_size: int,
                            data_ptr: int, data_size: int) -> None:
            func_name = self.runtime.read_from_memory(func_name_ptr, func_name_size).decode()
            print(func_name)
            func = remote_functions[func_name]
            data = self.runtime.read_from_memory(data_ptr, data_size)
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
            self._runtime.write_to_memory(buf_ptr, os.urandom(size))
            return WasiErrno.SUCCESS

        return random_get
