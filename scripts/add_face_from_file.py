import os
import sys

import cv2
import faiss
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))

from face_alignment import FaceAligment
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor


def box_size(bbox):
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    return bbox_w * bbox_h


def get_feature_vector(face_detector, feature_extractor, face_alignment, image_path):
    bboxes, scores, landmarks, image_origin = face_detector(image_path)

    # Get biggest bounding box
    bboxes_size = [box_size(bbox) for bbox in bboxes]
    idx_bbox = max(enumerate(bboxes_size), key=lambda x: x[1])[0]
    bbox, score, landmark = bboxes[idx_bbox], scores[idx_bbox], landmarks[idx_bbox]

    # Increase size of bbox
    bbox[0] = max(0, bbox[0] - int((bbox[2] - bbox[0]) * 0.15))
    bbox[1] = max(0, bbox[1] - int((bbox[3] - bbox[1]) * 0.15))
    bbox[2] = min(bbox[2] + int((bbox[2] - bbox[0]) * 0.15), image_origin.shape[1])
    bbox[3] = min(bbox[3] + int((bbox[3] - bbox[1]) * 0.15), image_origin.shape[0])

    # Crop box
    image_crop = image_origin.copy()[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]
    landmark[0::2] -= int(bbox[0])
    landmark[1::2] -= int(bbox[1])

    # Resize crop image to 112
    origin_shape = image_crop.shape[:2]
    target_shape = (112, 112)
    scale_ratio = max(target_shape[0] / origin_shape[0], target_shape[1] / origin_shape[1])
    image_crop = cv2.resize(image_crop, None, fx=scale_ratio, fy=scale_ratio)
    landmark[0::2] *= image_crop.shape[1] / origin_shape[1]
    landmark[1::2] *= image_crop.shape[0] / origin_shape[0]

    # Aligment
    image_align = face_alignment(image_crop, landmark)

    # Extract feature
    return feature_extractor(image_align)


def main(image_path: str, image_name: str, faiss_path: str, label_path: str):
    assert os.path.exists(image_path), image_path

    face_detector = FaceDetector(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "samples",
            "engines",
            "Primary_Detector",
            "yolov5_6_n_fa_widerface_640_1_fp16_simplify_dynamic_1_16_0.2_0.6_100_nms_0807_193941.trt",
        )
    )

    feature_extractor = FeatureExtractor(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "samples",
            "engines",
            "Secondary_Recognition",
            "default_ms1mv2_112x112_8_fp16_simplify_dynamic_1_64.trt",
        )
    )

    feace_aligment = FaceAligment()

    feature_vector = get_feature_vector(
        face_detector, feature_extractor, feace_aligment, image_path
    )

    # print(feature_vector)

    try:
        index = faiss.read_index(faiss_path)
        labels = [x.replace("\n", "") for x in open(label_path, "r").readlines()]
    except:
        index = faiss.IndexFlatIP(feature_extractor.feature_dim)
        labels = []

    print("index.is_trained", index.is_trained)

    index.add(np.expand_dims(feature_vector, axis=0))
    labels.append(image_name)

    print("index.ntotal", index.ntotal)

    faiss.write_index(index, faiss_path)
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")

    # D, I = index.search(np.repeat(np.expand_dims(feature_vector, axis=0), 2, axis=0), 2)

    # print(D)

    # print(I)


if __name__ == "__main__":
    assert len(sys.argv) == 3

    main(
        sys.argv[1],
        sys.argv[2],
        "faiss.index",
        "labels.txt",
    )