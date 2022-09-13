import os
import sys
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(
    engine,
    binding_to_type={"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32},
    max_batch_size=-1,
):
    """Allocates host and device buffer for TRT engine inference.
    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]

    image_size = None
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            image_size = engine.get_binding_shape(binding)[-2:]
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream, image_size


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class TRTInference(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, trt_engine_path, max_batch_size):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
        """
        self.max_batch_size = max_batch_size

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")

        # If we get here, the file with engine exists, so we can load it
        print("Loading cached TensorRT engine from {}".format(trt_engine_path))
        self.trt_engine = load_engine(self.trt_runtime, trt_engine_path)

        self.binding_to_type = dict()
        for index in range(self.trt_engine.num_bindings):
            name = self.trt_engine.get_binding_name(index)
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(index))
            shape = tuple(self.trt_engine.get_binding_shape(index))
            shape = list(map(lambda x: 1 if x == -1 else x, shape))
            # data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.binding_to_type[name] = dtype

        # This allocates memory for network inputs/outputs on both CPU and GPU
        (self.inputs, self.outputs, self.bindings, self.stream, self.image_size) = allocate_buffers(
            self.trt_engine,
            self.binding_to_type,
            self.max_batch_size,  # for dynamic shapes
        )

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

    def __call__(self, img, batch_size):
        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        if self.max_batch_size == -1:
            # Dynamic
            self.context.set_binding_shape(
                0, (batch_size, 3, self.image_size[0], self.image_size[0])
            )

        # Fetch output from the model
        outputs = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=batch_size,
        )

        # Output inference time
        print(
            "TensorRT inference time: {} ms".format(
                int(round((time.time() - inference_start_time) * 1000))
            )
        )

        # And return results
        return outputs
