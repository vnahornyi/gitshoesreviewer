import argparse

import onnx
from onnxconverter_common import float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Store ONNX weights and activations in float16, keeping float32 inputs and outputs")
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()

    model = onnx.load(args.source)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, args.target)
    print(f"wrote {args.target}")


if __name__ == "__main__":
    main()
