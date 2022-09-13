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


class FeatureExtractor(object):
    def __init__(self, trt_engine_path: str):
        self.batch_size = 1
        self.feature_dim = 512

        self.model = TRTInference(trt_engine_path, -1)

        self.image_size = self.model.image_size

        print(self.image_size)

    def _resize(image):
        return cv2.resize(image, (self.image_size[1], self.image_size[0]))

    def _transform(self, image):
        image = np.ascontiguousarray(image.transpose((2, 0, 1)).astype(np.float32)) / 255.0

        image = (image - 0.5) / 0.5

        input_array = np.expand_dims(image, axis=0)

        return input_array

    def __call__(self, image):
        feature_vector = self.model(self._transform(image), self.batch_size)[0]

        feature_vector = feature_vector / (
            np.sqrt(np.sum(np.square(feature_vector), axis=-1)) + 1e-8
        )

        return feature_vector


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "..", "docs", "A.png")
    bbox = [361.6000061, 271.40000153, 494.3999939, 434.40000916]
    bbox = [int(x) for x in bbox]

    image_origin = cv2.imread(image_path)
    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    image_crop = image_origin[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

    feature_extractor = FeatureExtractor(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "samples",
            "engines",
            "Secondary_Recognition",
            "webface_r50.trt",
        )
    )

    print(feature_extractor(feature_extractor._resize(image_crop)))
