import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from add_face_from_file import main

IMG_FORMATS = [
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".dng",
]


def is_image(path: str):
    return os.path.splitext(path)[1].lower() in IMG_FORMATS


if __name__ == "__main__":
    assert len(sys.argv) == 2
    folder_path = sys.argv[1]

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if is_image(file_path):
            main(file_path, os.path.splitext(file)[0], "faiss.index", "labels.txt")
