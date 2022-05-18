# snd4onnx
Simple node deletion tool for onnx. I only test very miscellaneous and limited patterns as a hobby. There are probably a large number of bugs. Pull requests are welcome.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/snd4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/snd4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/snd4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/snd4onnx?color=2BAF2B)](https://pypi.org/project/snd4onnx/) [![CodeQL](https://github.com/PINTO0309/snd4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/snd4onnx/actions?query=workflow%3ACodeQL)

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U snd4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ snd4onnx -h

usage:
    snd4onnx [-h]
    --remove_node_names REMOVE_NODE_NAMES [REMOVE_NODE_NAMES ...]
    --input_onnx_file_path INPUT_ONNX_FILE_PATH
    --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
    [--non_verbose]

optional arguments:
  -h, --help
        show this help message and exit.

  --remove_node_names REMOVE_NODE_NAMES [REMOVE_NODE_NAMES ...]
        ONNX node name to be deleted.

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
        Input onnx file path.

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
        Output onnx file path.

  --non_verbose
        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
>>> from snd4onnx import remove
>>> help(remove)

Help on function remove in module snd4onnx.onnx_remove_node:

remove(
    remove_node_names: List[str],
    input_onnx_file_path: Union[str, NoneType] = '',
    output_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
    non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    remove_node_names: List[str]
        List of OP names to be deleted.
        e.g. remove_node_names = ['op_name1', 'op_name2', 'op_name3', ...]

    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    removed_graph: onnx.ModelProto
        OP removed onnx ModelProto.
```

## 4. CLI Execution
```bash
$ snd4onnx \
--remove_node_names node_name_a node_name_b
--input_onnx_file_path input.onnx \
--output_onnx_file_path output.onnx
```

## 5. In-script Execution
```python
from snd4onnx import remove

onnx_graph = remove(
    remove_node_names=['node_name_a', 'node_name_b'],
    input_onnx_file_path='input.onnx',
)

# or

onnx_graph = remove(
    remove_node_names=['node_name_a', 'node_name_b'],
    onnx_graph=graph,
)
```

## 6. Sample
### 6-1. sample.1
|Before|After|
|:-:|:-:|
|![test1 onnx](https://user-images.githubusercontent.com/33194443/161254346-cdcf861f-adf6-447e-8a8b-3abe619bf5ee.png)|![test1_removed onnx](https://user-images.githubusercontent.com/33194443/161254523-7f9d2f76-51ea-440d-a06b-7cda475a059d.png)|
### 6-2. sample.2
|Before|After|
|:-:|:-:|
|![test3 onnx](https://user-images.githubusercontent.com/33194443/161255204-6412469d-68f9-4e92-8cdd-2d6c1ca16b39.png)|![test3_removed onnx](https://user-images.githubusercontent.com/33194443/161255237-24e48064-795f-4ed3-bd31-9ba50b58de93.png)|
### 6-3. sample.3
|Before|After|
|:-:|:-:|
|![test5 onnx](https://user-images.githubusercontent.com/33194443/161255498-148ab730-bdcc-4140-97fc-010aff0550ef.png)|![test5_removed onnx](https://user-images.githubusercontent.com/33194443/161255532-13d2bfbb-7051-4c46-8025-1e2b6e2c61c5.png)|
### 6-4. sample.4
|Before|After|
|:-:|:-:|
|![test7 onnx](https://user-images.githubusercontent.com/33194443/161255804-c088a069-c049-4b4b-9e01-1827df9746c5.png)|![test7_removed onnx](https://user-images.githubusercontent.com/33194443/161255996-155eb870-52d7-4694-b2b9-d524d996a671.png)|
### 6-5. sample.5
|Before|After|
|:-:|:-:|
|![test8 onnx](https://user-images.githubusercontent.com/33194443/161256392-d557322d-b358-4949-bd66-f5e678d131dc.png)|![test8_removed onnx](https://user-images.githubusercontent.com/33194443/161256404-8e20596f-c7c2-4da3-a6b4-9685eda32ff8.png)|

## 7. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
3. https://github.com/PINTO0309/scs4onnx
4. https://github.com/PINTO0309/sne4onnx
5. https://github.com/PINTO0309/snc4onnx
6. https://github.com/PINTO0309/sog4onnx
7. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
