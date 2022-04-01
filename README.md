# snd4onnx
Simple node deletion tool for onnx.

## 1. Setup
```bash
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U snd4onnx
```

## 2. Usage
```bash
$ snd4onnx -h

usage: onnx_remove_node.py [-h] onnx_file_path remove_node_names

positional arguments:
  onnx_file_path
  remove_node_names

optional arguments:
  -h, --help         show this help message and exit
```

## 3. Execution
```bash
$ snd4onnx input.onnx node_name_a
```

## 4. Sample


## 5. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon