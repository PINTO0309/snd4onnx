# snd4onnx
Simple node deletion tool for onnx.

## 1. Setup
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
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
### 4-1. sample.1
|Before|After|
|:-:|:-:|
|![test1 onnx](https://user-images.githubusercontent.com/33194443/161254346-cdcf861f-adf6-447e-8a8b-3abe619bf5ee.png)|![test1_removed onnx](https://user-images.githubusercontent.com/33194443/161254523-7f9d2f76-51ea-440d-a06b-7cda475a059d.png)|
### 4-2. sample.2
|Before|After|
|:-:|:-:|
|![test3 onnx](https://user-images.githubusercontent.com/33194443/161255204-6412469d-68f9-4e92-8cdd-2d6c1ca16b39.png)|![test3_removed onnx](https://user-images.githubusercontent.com/33194443/161255237-24e48064-795f-4ed3-bd31-9ba50b58de93.png)|
### 4-3. sample.3
|Before|After|
|:-:|:-:|
|![test5 onnx](https://user-images.githubusercontent.com/33194443/161255498-148ab730-bdcc-4140-97fc-010aff0550ef.png)|![test5_removed onnx](https://user-images.githubusercontent.com/33194443/161255532-13d2bfbb-7051-4c46-8025-1e2b6e2c61c5.png)|
### 4-4. sample.4
|Before|After|
|:-:|:-:|
|![test7 onnx](https://user-images.githubusercontent.com/33194443/161255804-c088a069-c049-4b4b-9e01-1827df9746c5.png)|![test7_removed onnx](https://user-images.githubusercontent.com/33194443/161255996-155eb870-52d7-4694-b2b9-d524d996a671.png)|
### 4-5. sample.5
|Before|After|
|:-:|:-:|
|![test8 onnx](https://user-images.githubusercontent.com/33194443/161256392-d557322d-b358-4949-bd66-f5e678d131dc.png)|![test8_removed onnx](https://user-images.githubusercontent.com/33194443/161256404-8e20596f-c7c2-4da3-a6b4-9685eda32ff8.png)|

## 5. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
