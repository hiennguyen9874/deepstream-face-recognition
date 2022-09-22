import cv2
import matplotlib.pyplot as plt

BOX_COLOR = (255, 0, 0)
LEFT_COLOR = (255, 0, 0)
CENTER_COLOR = (0, 255, 0)
RIGHT_COLOR = (0, 0, 255)


__all__ = ["visualize"]


def visualize_bbox(
    img,
    bbox,
    landmark,
    landmark_mask,
    color=BOX_COLOR,
    thickness=2,
    radius=2,
    bbox_type="coco",
    landmark_normalized: bool = True,
):
    r"""Visualizes a single bounding box on the image
    Args:
        bbox: coco
    """

    if bbox_type == "coco":
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = (
            int(x_min),
            int(x_min + w),
            int(y_min),
            int(y_min + h),
        )
    elif bbox_type == "pascal_voc":
        x_min, y_min, x_max, y_max = bbox
    elif bbox_type == "albumentations":
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = (
            int(x_min * img.shape[1]),
            int(y_min * img.shape[0]),
            int(x_max * img.shape[1]),
            int(y_max * img.shape[0]),
        )
    else:
        raise KeyError("bbox_type error")

    x_min, y_min, x_max, y_max = (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max)),
        int(round(y_max)),
    )

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    if landmark is not None and landmark_mask > 0:
        img = cv2.circle(
            img,
            (int(landmark[0] * img.shape[1]), int(landmark[1] * img.shape[0]))
            if landmark_normalized
            else (int(landmark[0]), int(landmark[1])),
            radius=radius,
            color=LEFT_COLOR,
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(landmark[2] * img.shape[1]), int(landmark[3] * img.shape[0]))
            if landmark_normalized
            else (int(landmark[2]), int(landmark[3])),
            radius=radius,
            color=RIGHT_COLOR,
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(landmark[4] * img.shape[1]), int(landmark[5] * img.shape[0]))
            if landmark_normalized
            else (int(landmark[4]), int(landmark[5])),
            radius=radius,
            color=CENTER_COLOR,
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(landmark[6] * img.shape[1]), int(landmark[7] * img.shape[0]))
            if landmark_normalized
            else (int(landmark[6]), int(landmark[7])),
            radius=radius,
            color=LEFT_COLOR,
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(landmark[8] * img.shape[1]), int(landmark[9] * img.shape[0]))
            if landmark_normalized
            else (int(landmark[8]), int(landmark[9])),
            radius=radius,
            color=RIGHT_COLOR,
            thickness=-1,
        )

    return img


def visualize(
    image,
    bboxes,
    landmarks,
    landmarks_mask,
    bbox_type="coco",
    thickness=2,
    radius=2,
    landmark_normalized: bool = True,
):
    img = image.copy()
    for bbox, landmark, landmark_mask in zip(bboxes, landmarks, landmarks_mask):
        img = visualize_bbox(
            img,
            bbox,
            landmark,
            landmark_mask,
            bbox_type=bbox_type,
            thickness=thickness,
            radius=radius,
            landmark_normalized=landmark_normalized,
        )
    return img
