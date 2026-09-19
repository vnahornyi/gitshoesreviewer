import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed


def fix_shapes(model: onnx.ModelProto) -> None:
    # Core ML compiles a static graph; the RTMPose exports leave the batch and joint dimensions open.
    make_dim_param_fixed(model.graph, "batch", 1)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {i.name: np.zeros(i.shape, np.float32) for i in session.get_inputs()}
    for output, value in zip(model.graph.output, session.run(None, feeds)):
        dims = output.type.tensor_type.shape.dim
        for dim, size in zip(dims, value.shape):
            dim.Clear()
            dim.dim_value = size


def main() -> None:
    parser = argparse.ArgumentParser(description="Store ONNX weights and activations in float16, keeping float32 inputs and outputs")
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()

    model = onnx.load(args.source)
    fix_shapes(model)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, args.target)
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
