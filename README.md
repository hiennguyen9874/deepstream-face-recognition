# Deepstream face detection & recognition

Demo output video: [![Deepstream face detection & recognition](https://img.youtube.com/vi/eim-uTRNYtg/0.jpg)](https://www.youtube.com/watch?v=eim-uTRNYtg)

This docs for dGPU, for jetson using [./JETSON.md](./JETSON.md)

## Export `.engine`

### Face detection model

-   Export checkpoint to onnx model:
    -   Clone [yolov7-face-detection](https://github.com/hiennguyen9874/yolov7-face-detection/tree/using-landmark) and cd into `yolov7-face-detection` folder
    -   Download weight and save into `weights/yolov7-tiny33.pt`
    -   Export to onnx: `python3 export.py --weights ./weights/yolov7-tiny33.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup --trt`
-   Or download onnx file from from [github.com/hiennguyen9874/yolov7-face-detection/releases/tag/v0.1](https://github.com/hiennguyen9874/yolov7-face-detection/releases/tag/v0.1)
-   Export to TensorRT: `/usr/src/tensorrt/bin/trtexec --onnx=samples/models/Primary_Detector/yolov7-tiny41-nms-trt.onnx --saveEngine=samples/engines/Primary_Detector/yolov7-tiny41-nms-trt.trt --workspace=14336 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640`

### Face recognition model

-   Download `webface_r50.onnx` from [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) and cleaning onnx file: `python3 scripts/onnx_clean.py --onnx-path samples/models/Secondary_Recognition/webface_r50.onnx --image-size 112,112 --batch-size 1 --simplify --dynamic --cleanup --add-norm`
-   Or download onnx file from from: [github.com/hiennguyen9874/deepstream-face-recognition/releases/tag/v0.1](https://github.com/hiennguyen9874/deepstream-face-recognition/releases/tag/v0.1)
-   Export to TensorRT: `/usr/src/tensorrt/bin/trtexec --onnx=samples/models/Secondary_Recognition/webface_r50_norm_dynamic_simplify_cleanup.onnx --saveEngine=samples/engines/Secondary_Recognition/webface_r50_norm_dynamic_simplify_cleanup.trt --workspace=14336 --fp16 --minShapes=input.1:1x3x112x112 --optShapes=input.1:4x3x112x112 --maxShapes=input.1:16x3x112x112 --shapes=input.1:4x3x112x112`

### Download demo video

-   Download [Friends.mp4](https://github.com/hiennguyen9874/deepstream-face-recognition/releases/download/v0.1/Friends.mp4) or other video and save into samples/videos/
-   Modify `uri` in [./samples/configs/deepstream_app.txt](./samples/configs/deepstream_app.txt)

## Docker

### Prerequisites

-   Docker
-   Nvidia-driver
-   Nvidia-docker2

### Usage

-   Pull image: `docker pull hiennguyen9874/deepstream-face-recognition:deepstream-6.0.1`
-   Run bash inside docker: `docker run --runtime nvidia --rm -it -v $(pwd):/app hiennguyen9874/deepstream-face-recognition:deepstream-6.0.1`
    -   Add new face:
        -   A: `python3 scripts/add_face_from_file.py A docs/A.png docs/A2.png`
        -   B: `python3 scripts/add_face_from_file.py B docs/B.png docs/B2.png`
        -   C: `python3 scripts/add_face_from_file.py C docs/C.png docs/C2.png`
    -   Build source: `bash sources/install.sh`
    -   Run: `./bin/deepstream-app -c samples/configs/deepstream_app.txt`
    -   Output in [./outputs/videos](./outputs/videos/)

## Without docker

### Prerequisites

-   Deepstream 6.0.1
-   TensorRT & python tensorrt
-   [github.com/hiennguyen9874/TensorRT](https://github.com/hiennguyen9874/TensorRT)
-   [Faiss](./FAISS.md)
-   python package: `pip3 install opencv-python numpy scikit-image pycuda pillow matplotlib`

### Add new face

-   A: `python3 scripts/add_face_from_file.py A docs/A.png docs/A2.png`
-   B: `python3 scripts/add_face_from_file.py B docs/B.png docs/B2.png`
-   C: `python3 scripts/add_face_from_file.py C docs/C.png docs/C2.png`

### Build

-   `sudo bash sources/install.sh`

### Run

-   `./bin/deepstream-app -c samples/configs/deepstream_app.txt`
