# import onnxruntime as ort
import numpy as np
import time, os
from onnxruntime_extensions import PyOp, onnx_op, PyOrtFunction
from finn.core.datatype import DataType
from driver_base import FINNExampleOverlay
from dataset_loading import mnist

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
bsize = 1000
bitfile = "resizer.bit"
outputfile = "output.npy"
runtime_weight_dir = "runtime_weights/"

accel = FINNExampleOverlay(
    bitfile_name=bitfile,
    platform=platform,
    io_shape_dict=io_shape_dict,
    batch_size=bsize,
    runtime_weight_dir="runtime_weights/",
)

# Implement the CustomOp by decorating a function with onnx_op
@onnx_op(op_type="StreamingDataflowPartition", inputs=[PyOp.dt_float], outputs=[PyOp.dt_int64])
def StreamingDataflowPartition(inputs):
    obuf_normal = accel.execute(inputs)
    # np.save(outputfile, obuf_normal)
    return np.int64(obuf_normal)


trainx, trainy, test_imgs, test_labels, valx, valy = mnist.load_mnist_data(
    "/tmp", download=True, one_hot=False
)

ok = 0
nok = 0
total = test_imgs.shape[0]

n_batches = int(total / bsize)

test_imgs = np.transpose(test_imgs, (0, 3, 1, 2)).astype(np.float32)
test_labels = test_labels.reshape(n_batches, bsize)

model_func = PyOrtFunction.from_model("./tfc_w1_a1_dataflow_parent_1000.onnx")

start_time = time.time()

for i in range(n_batches):
        test_img = test_imgs[range(i*bsize, (i+1)*bsize), :, :, :]
        exp = test_labels[i]

        results = model_func(test_img)
        
        ret = np.bincount(results.flatten() == exp.flatten())
        nok += ret[0]
        ok += ret[1]
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

end_time = time.time()
runtime = end_time - start_time
acc = 100.0 * ok / (total)
print("Accuracy: %f" % acc)
print("Runtime (s): {}".format(runtime))