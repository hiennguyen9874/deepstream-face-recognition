import cv2
import numpy as np
from skimage import transform as trans


class FaceAligment(object):
    output_size = (112, 112)
    default_array = np.array(
        [
            [38.29459953 + 8.0, 51.69630051],
            [73.53179932 + 8.0, 51.50139999],
            [56.02519989 + 8.0, 71.73660278],
            [41.54930115 + 8.0, 92.3655014],
            [70.72990036 + 8.0, 92.20410156],
        ],
        dtype=np.float32,
    )

    def __call__(self, image, landmarks):
        landmarks = np.array(landmarks, dtype=np.float32).reshape(5, 2)

        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.default_array)
        tfm = tform.params[0:2, :]

        return cv2.warpAffine(image, tfm, (self.output_size[0], self.output_size[1]))
