## Export

### Export yolo face detection model

-   `python3 src/export_trt_batched_nms.py --cfg-source configs/sources/widerface.yaml --cfg-model configs/models/yolov5_6_n_fa.yaml --checkpoint saved/checkpoints/0807_193941/val_best_map_fitness.pth --image-size 640 --batch-size 1 --simplify --device 1 --dynamic --conf-thres 0.2 --iou-thres 0.5 --run-id 0807_193941 --min-batch-size 1 --max-batch-size 16 --half`

-   `python3 src/export_trt_batched_nms.py --cfg-source configs/sources/widerface.yaml --cfg-model configs/models/yolov5_6_n_ws.yaml --checkpoint saved/checkpoints/0807_173751/val_best_map_fitness.pth --image-size 640 --batch-size 1 --simplify --device 1 --dynamic --conf-thres 0.2 --iou-thres 0.5 --run-id 0807_173751 --min-batch-size 1 --max-batch-size 16 --half`

### Export face recognition model

-   `python3 src/export_trt.py --cfg-source configs/source/ms1mv2.yml --cfg-model configs/model/default.yml --checkpoint-path saved/checkpoints/0805_100219/val_best_accuracy_agedb_30.pth --image-size 112,112 --batch-size 8 --min-batch-size 1 --max-batch-size 64 --half --simplify --dynamic --device 0`

## Add new face

## Test
