# Deepstream face recognition

## Prerequisites

-   Deepstream 6.0.1
-   [github.com/hiennguyen9874/TensorRT](https://github.com/hiennguyen9874/TensorRT)
-   [Faiss](./FAISS.md)

## Export `.engine`

### Face detection model

-   Clone [yolov7-face-detection](https://github.com/hiennguyen9874/yolov7-face-detection) and cd into `yolov7-face-detection` folder
-   Download weight and save into `weights/yolov7-tiny33.pt`
-   Export to onnx: `python3 export.py --weights ./weights/yolov7-tiny33.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup --trt`
-   Export to TensorRT: `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-tiny33.onnx --saveEngine=./weights/yolov7-tiny33-nms-trt.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640`

### Face recognition model

-   Download `webface_r50.onnx` from [deepinsight/insightface](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
-   Export onnx to tensorRT: `python3 scripts/onnx2tensorrt.py --onnx-path samples/models/Secondary_Recognition/ms1mv3_r50.onnx --image-size 112,112 --batch-size 8 --min-batch-size 1 --max-batch-size 64 --half --simplify --dynamic --cleanup --file-save samples/engines/Secondary_Recognition/webface_r50.trt`

## Add new face

-   A: `python3 scripts/add_face_from_file.py A docs/A.png docs/A2.png`
-   B: `python3 scripts/add_face_from_file.py B docs/B.png docs/B2.png`
-   C: `python3 scripts/add_face_from_file.py C docs/C.png docs/C2.png`

## Test

-   `bash ./deepstream-app.sh`
