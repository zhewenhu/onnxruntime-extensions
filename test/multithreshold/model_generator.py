import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
v = helper.make_tensor_value_info('v', TensorProto.DOUBLE, [6, 3, 2, 2])
thresholds = helper.make_tensor_value_info('thresholds', TensorProto.DOUBLE, [3, 7])

# Create one output (ValueInfoProto)
results = helper.make_tensor_value_info('results', TensorProto.DOUBLE, [6, 3, 2, 2])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
        "MultiThreshold",
        ["v", "thresholds"],
        ["results"],
        domain="ai.onnx.contrib",
        out_dtype="",
        out_scale=2.0,
        out_bias=-1.0,
        data_layout="NCHW",
    )

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # nodes
    'test-model',      # name
    [v, thresholds],   # inputs
    [results],         # outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='test')

print('The model is:\n{}'.format(model_def))

onnx.save(model_def, 'test.onnx')
