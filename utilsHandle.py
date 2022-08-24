import os
import sys
from pathlib import Path
from os.path import join
from os import listdir

dir_path = os.path.dirname(os.path.realpath(__file__))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

max_length, attention_features_shape = 50, 64


class pathImg:
    def __init__(self):
        path_card = ''
        path_avatar = ''

    def set_path_card(self, path):
        self.path_Card = path

    def set_path_avatar(self, path):
        self.path_avatar = path


pathImg = pathImg()


def get_path():
    return pathImg

# pathImg.set_path_avatar(join(path_folder, imgPath))
# pathImg.set_path_card(join(path_folder, imgPath))


def predict(source):
    result = "ok"
    print(result)
    return result

