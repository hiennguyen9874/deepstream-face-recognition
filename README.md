## Export yolo face detection model

-   `python3 src/export_trt.py --cfg-source configs/sources/widerface.yaml --cfg-model configs/models/yolov5_6_n_fa.yaml --checkpoint-path saved/checkpoints/0807_193941/val_best_map_fitness.pth --image-size 640 --batch-size 1 --min-batch-size 1 --max-batch-size 16 --half --simplify --dynamic --device 1 --run-id 0807_193941`

## Export face recognition model
