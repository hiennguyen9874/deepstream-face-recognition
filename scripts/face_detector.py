import os
import sys
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__)))

from trt_inference import TRTInference


class FaceDetector(object):
    def __init__(self, trt_engine_path: str):
        self.batch_size = 1

        self.model = TRTInference(trt_engine_path, -1)

        self.image_size = self.model.image_size

    def _imread(self, image_path):
        return cv2.imread(image_path)

    def _bgr2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _resize_padding(
        self,
        image,
        target_shape,
        fit_mode: str = "fit",
        rect_crop: bool = False,
        stride: int = 32,
        color=(114, 114, 114),
    ):
        # boxes must be yolo format
        assert fit_mode in {"fit", "stretch", "center"}
        if fit_mode == "stretch" and rect_crop:
            raise ValueError("stretch mode not work with rect_crop")

        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)
        origin_shape = (image.shape[0], image.shape[1])

        if fit_mode == "stretch":
            return cv2.resize(image, target_shape[::-1], interpolation=cv2.INTER_LINEAR), {
                "origin_shape": target_shape,
                "unpad_shape": target_shape,
                "after_shape": target_shape,
                "padding": (0, 0),
            }

        if fit_mode == "fit":
            scale_ratio = min(target_shape[0] / origin_shape[0], target_shape[1] / origin_shape[1])
        else:
            scale_ratio = min(
                target_shape[0] / origin_shape[0], target_shape[1] / origin_shape[1], 1
            )

        unpad_shape = int(round(origin_shape[0] * scale_ratio)), int(
            round(origin_shape[1] * scale_ratio)
        )

        padding_height, padding_width = (
            target_shape[0] - unpad_shape[0],
            target_shape[1] - unpad_shape[1],
        )

        if rect_crop:
            padding_height, padding_width = (
                padding_height % stride,
                padding_width % stride,
            )

        padding_height, padding_width = padding_height / 2, padding_width / 2

        image = cv2.resize(image, unpad_shape[::-1], interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(padding_height - 0.1)), int(round(padding_height + 0.1))
        left, right = int(round(padding_width - 0.1)), int(round(padding_width + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return image, {
            "before_shape": origin_shape,
            "after_shape": image.shape[:2],
            "padding": (padding_height, padding_width),
        }

    def _preprocess(self, image_origin):
        image, shape_info = self._resize_padding(
            image_origin,
            fit_mode="center",
            target_shape=self.image_size,
            rect_crop=False,
        )

        # cv2.imwrite("debug.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        input_array = np.ascontiguousarray(image.transpose((2, 0, 1)).astype(np.float32)) / 255.0
        input_array = np.expand_dims(input_array, axis=0)

        return input_array, shape_info

    def cxcy_to_xy(self, bboxes):
        r"""Convert bboxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
        Args:
            bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)
        """
        ret = bboxes.copy()

        ret[..., :2] = bboxes[..., :2] - (bboxes[..., 2:] / 2)
        ret[..., 2:] = bboxes[..., :2] + (bboxes[..., 2:] / 2)

        return ret

    def xy_to_cxcy(self, bboxes):
        r"""Convert bboxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        Args:
            bboxes: Bounding boxes, a tensor of dimensions (n_object, 4)
        Out:
            bboxes in center coordinate
        """
        ret = bboxes.copy()

        ret[..., :2] = (bboxes[..., 2:] + bboxes[..., :2]) / 2
        ret[..., 2:] = bboxes[..., 2:] - bboxes[..., :2]

        return ret

    def remove_scale_padding(
        self,
        bboxes,
        before_shape,
        after_shape,
        padding,
        bbox_in_format: str = "yolo",
        bbox_out_format: str = "yolo",
    ):
        r"""Inverse version of add_scale_padding

        Args:
            bboxes ([type]): yolo format bounding box before remove (scaling and padding)
            before_shape (Tuple[height: int, width: int]): shape of image before remove (scaling and padding)
            after_shape (Tuple[height: int, width: int]): shape of image before remove (scaling and padding)
            padding (Tuple[height: int, width: int]): padding width and height size
        Returns:
            (yolo format): bounding box after remove (scaling and padding)
        """

        assert bbox_in_format in {
            "yolo",
            "pascal-voc",
        }, "bbox format must in {yolo, pascal-voc}"

        assert bbox_out_format in {
            "yolo",
            "pascal-voc",
        }, "bbox format must in {yolo, pascal-voc}"

        if bbox_in_format == "yolo":
            ret = self.cxcy_to_xy(
                bboxes
                * np.array([before_shape[1], before_shape[0], before_shape[1], before_shape[0]])
            )
        else:
            ret = bboxes

        ret[:, 0] = (ret[:, 0] - padding[1]) / (before_shape[1] - padding[1] * 2)
        ret[:, 1] = (ret[:, 1] - padding[0]) / (before_shape[0] - padding[0] * 2)
        ret[:, 2] = (ret[:, 2] - padding[1]) / (before_shape[1] - padding[1] * 2)
        ret[:, 3] = (ret[:, 3] - padding[0]) / (before_shape[0] - padding[0] * 2)

        if bbox_out_format == "yolo":
            return self.xy_to_cxcy(ret)

        return ret * np.array([after_shape[1], after_shape[0], after_shape[1], after_shape[0]])

    def remove_scale_padding_landmark(self, landmarks, before_shape, after_shape, padding):
        r"""
        Args:
            bboxes ([type]): yolo format bounding box before remove (scaling and padding)
            before_shape (Tuple[height: int, width: int]): shape of image before remove (scaling and padding)
            after_shape (Tuple[height: int, width: int]): shape of image before remove (scaling and padding)
            padding (Tuple[height: int, width: int]): padding width and height size
        Returns:
            (yolo format): bounding box after remove (scaling and padding)
        """

        landmarks[:, 0::2] = (
            (landmarks[:, 0::2] - padding[1]) / (before_shape[1] - padding[1] * 2)
        ) * after_shape[1]

        landmarks[:, 1::2] = (
            (landmarks[:, 1::2] - padding[0]) / (before_shape[0] - padding[0] * 2)
        ) * after_shape[0]

        return landmarks

    def __call__(self, image_path):
        image_origin = self._bgr2rgb(self._imread(image_path))

        input_array, shape_info = self._preprocess(image_origin)

        result = self.model(input_array, self.batch_size)

        keepCount, bboxes, scores, classes, landmarks = result

        bboxes = bboxes[: int(keepCount) * 4].reshape(-1, 4)
        scores = scores[: int(keepCount)]
        landmarks = landmarks[: int(keepCount) * 11].reshape(-1, 11)[..., :10]

        bboxes = self.remove_scale_padding(
            bboxes,
            before_shape=shape_info["after_shape"],
            after_shape=shape_info["before_shape"],
            padding=shape_info["padding"],
            bbox_in_format="pascal-voc",
            bbox_out_format="pascal-voc",
        )

        landmarks = self.remove_scale_padding_landmark(
            landmarks,
            before_shape=shape_info["after_shape"],
            after_shape=shape_info["before_shape"],
            padding=shape_info["padding"],
        )

        return bboxes, scores, landmarks, image_origin


if __name__ == "__main__":
    face_detector = FaceDetector(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "samples",
            "engines",
            "Primary_Detector",
            "yolov7-tiny41-nms-trt.trt",
        )
    )

    print(face_detector(os.path.join(os.path.dirname(__file__), "..", "docs", "A.png"))[:3])
