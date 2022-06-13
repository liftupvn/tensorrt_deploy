
import os
import numpy as np
import trt
import cv2 
import time
TRT_ENGINE_PATH= "engine.engine"
ONNX_FILE_PATH= "end2end.onnx"
TRT_LOGGER = ""
def build_engine(onnx_file_path, save_engine=False):
    if os.path.exists(TRT_ENGINE_PATH):
        # If a serialized engine exists, you can use the existing serialized engine instead of creating a new one. 
        print("Reading engine from file {}".format(TRT_ENGINE_PATH))
        with open(TRT_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            return engine, context

    # Initialize the TensorRT engine and parse the ONNX model. 
    builder = trt.Builder(TRT_LOGGER)

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Specify that the TensorRT engine can use at most 1 GB of GPU memory for policy selection. 
    builder.max_workspace_size = 1 << 30
    # In this example, only one image is included in the batch process. 
    builder.max_batch_size = 1
    # We recommend that you use the FP16 mode. 
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # Parse the ONNX model. 
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # Create a TensorRT engine that is optimized for the platform on which the TensorRT engine is deployed. 
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    with open(TRT_ENGINE_PATH, "wb") as f:
        print("Save engine to {}".format(TRT_ENGINE_PATH))
        f.write(engine.serialize())

    return engine, context

engine, context = build_engine(ONNX_FILE_PATH)
# Obtain the input data size and output data size. Allocate memory to process the input data and output data based on your business requirements. 
for binding in engine:
    if engine.binding_is_input(binding):  # we expect only one input
        input_shape = engine.get_binding_shape(binding)
        input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
        device_input = cuda.mem_alloc(input_size)
    else:  # The output data. 
        output_shape = engine.get_binding_shape(binding)
        # Create a page-locked memory buffer. This way, the data is not written to the disk. 
        host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
        device_output = cuda.mem_alloc(host_output.nbytes)
# Create a stream, copy the input data and output data to the stream, and then run inference. 
# stream = cuda.Stream()
# Preprocess the input data. 
host_input = np.array(cv2.imread("car.jpg"), dtype=np.float32, order='C')
# cuda.memcpy_htod_async(device_input, host_input, stream)
# Run inference. 
start = time.time()
# context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
# cuda.memcpy_dtoh_async(host_output, device_output, stream)
# stream.synchronize()
cost = time.time() - start
print(f"tensorrt predict_cost = {cost}")
