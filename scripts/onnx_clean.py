import argparse
import os
import tempfile
from pathlib import Path

import cv2
import onnx
import onnxsim
import torch
from onnx.tools import update_model_dims


def onnx_get_input_output(onnx_path):
    model_onnx = onnx.load(onnx_path)

    output = [node.name for node in model_onnx.graph.output]

    input_all = [node.name for node in model_onnx.graph.input]
    input_initializer = [node.name for node in model_onnx.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    print("Inputs: ", net_feed_input)
    print("Outputs: ", output)

    return net_feed_input[0], output[0]


def onnx_dynamic(onnx_path, onnx_dynamic_path, intput_name, output_name):
    model_onnx = onnx.load(onnx_path)

    input_dims = {
        intput_name: ["b", 3, 112, 112],
    }

    output_dims = {output_name: ["b", 512]}

    updated_model = update_model_dims.update_inputs_outputs_dims(
        model_onnx, input_dims, output_dims
    )

    onnx.save(updated_model, onnx_dynamic_path)
    assert os.path.exists(onnx_dynamic_path)


def onnx_simplify(onnx_path, onnx_simplify_path, image, dynamic, input_name):
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={input_name: list(image.shape)} if dynamic else None,
    )
    onnx.save(model_onnx, onnx_simplify_path)

    assert os.path.exists(onnx_simplify_path)


def onnx_cleanup(onnx_path, onnx_cleanup_path):
    import onnx_graphsurgeon as gs

    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    graph = gs.import_onnx(model_onnx)
    graph = graph.cleanup().toposort()
    model_onnx = gs.export_onnx(graph)
    onnx.save(model_onnx, onnx_cleanup_path)

    assert os.path.exists(onnx_cleanup_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--onnx-path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="384,128",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
    )
    args = parser.parse_args()

    args.image_size = [int(x) for x in args.image_size.split(",")]

    image = torch.randn((args.batch_size, 3, args.image_size[0], args.image_size[1]))

    input_name, output_name = onnx_get_input_output(args.onnx_path)

    args.onnx_dynamic_path = f"{os.path.splitext(args.onnx_path)[0]}_dynamic.onnx"
    onnx_dynamic(args.onnx_path, args.onnx_dynamic_path, input_name, output_name)
    args.onnx_path = args.onnx_dynamic_path

    if args.simplify:
        args.onnx_simplify_path = f"{os.path.splitext(args.onnx_path)[0]}_simplify.onnx"
        onnx_simplify(args.onnx_path, args.onnx_simplify_path, image, args.dynamic, input_name)
        args.onnx_path = args.onnx_simplify_path

    if args.cleanup:
        args.onnx_cleanup_path = f"{os.path.splitext(args.onnx_path)[0]}_cleanup.onnx"
        onnx_cleanup(args.onnx_path, args.onnx_cleanup_path)
        args.onnx_path = args.onnx_cleanup_path
