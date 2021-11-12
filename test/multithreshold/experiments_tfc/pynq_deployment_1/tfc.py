# import onnxruntime as ort
import numpy as np
import os
from onnxruntime_extensions import get_library_path, PyOp, onnx_op, PyOrtFunction
from finn.core.datatype import DataType
from driver_base import FINNExampleOverlay

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : DataType.INT32,
    "odt" : DataType.UINT8,
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : (1, 64),
    "oshape_normal" : (1, 1),
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : (1, 64, 1),
    "oshape_folded" : (1, 1, 1),
    "ishape_packed" : (1, 64, 4),
    "oshape_packed" : (1, 1, 1),
    "input_dma_name" : 'idma0',
    "number_of_external_weights": 0
}
platform = "zynq-iodma"
batch_size = 1
bitfile = "resizer.bit"
outputfile = "output.npy"
runtime_weight_dir = "runtime_weights/"

# instantiate FINN accelerator driver and pass batchsize and bitfile
accel = FINNExampleOverlay(
    bitfile_name = bitfile, platform = platform,
    io_shape_dict = io_shape_dict, batch_size = batch_size,
    runtime_weight_dir = runtime_weight_dir
)

# Implement the CustomOp by decorating a function with onnx_op
@onnx_op(op_type="StreamingDataflowPartition", inputs=[PyOp.dt_float], outputs=[PyOp.dt_int64])
def StreamingDataflowPartition(inputs):
    obuf_normal = accel.execute(inputs)
    # np.save(outputfile, obuf_normal)
    return np.int64(obuf_normal)

inputs = np.load("input.npy")

model_func = PyOrtFunction.from_model("./tfc_w1_a1_dataflow_parent.onnx")
outputs = model_func(inputs)
print(outputs)
