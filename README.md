## Export yolo face detection model

-   `python3 src/export_trt.py --cfg-source configs/sources/widerface.yaml --cfg-model configs/models/yolov5_6_n_fa.yaml --checkpoint-path saved/checkpoints/0807_001907/last.pth --image-size 640 --batch-size 1 --min-batch-size 1 --max-batch-size 16 --half --simplify --dynamic --device 1 --run-id 0807_001907`
-   `python3 src/export_trt.py --cfg-source configs/sources/widerface.yaml --cfg-model configs/models/yolov5_6_n_ws.yaml --checkpoint-path saved/checkpoints/0806_212222/last.pth --image-size 640 --batch-size 1 --min-batch-size 1 --max-batch-size 16 --half --simplify --dynamic --device 1 --run-id 0806_212222`
