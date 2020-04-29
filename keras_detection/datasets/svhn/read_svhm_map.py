import json
import pickle
from glob import glob
from json import JSONEncoder
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from keras_detection import Features, LabelsFrame, ImageData


def load_dataset(dataset_dir: Path, as_numpy: bool = False):
    dataset_dir = Path(dataset_dir)

    train_annotations_file = dataset_dir / 'digitStruct.json'
    digits_struct_file = dataset_dir / 'digitStruct.mat'
    dataset_pkl_file = dataset_dir / (dataset_dir.name + ".pkl")

    if dataset_pkl_file.exists():
        dataset_np = pickle.load(dataset_pkl_file.open("rb"))
        print(f"Loaded {dataset_pkl_file.name} dataset from existing pickle file. N={len(dataset_np)}")
    else:
        if not digits_struct_file.exists():
            raise FileExistsError(f"Cannot open {digits_struct_file}")

        if not train_annotations_file.exists():
            print(f"Parsing {digits_struct_file}, this may take a while")
            dsf = DigitStructFile(str(digits_struct_file))
            annotations = dsf.load()
            fout = open(train_annotations_file, 'w')
            fout.write(JSONEncoder(indent=True).encode(annotations))
            fout.close()
        else:
            print(f"Loading existing JSON annotations from {train_annotations_file}")
            with open(train_annotations_file, 'r') as file:
                annotations = json.load(file)

        print(f"Loaded annotations for {len(annotations)} images")
        print(f"Loading images ... ")
        images = {
            Path(p).name: np.array(Image.open(p))
            for p in tqdm(glob(f"{dataset_dir}/*.png"))
        }
        dataset_np = prepare_dataset(images, annotations, dataset_pkl_file)

    num_examples = len(dataset_np)
    if as_numpy:
        return dataset_np, num_examples

    def dataset_generator(svhn_dataset):
        for element in svhn_dataset:
            yield element

    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(dataset_np),
        ImageData.dataset_dtypes(), ImageData.dataset_shapes()
    )
    return dataset, num_examples


def prepare_dataset(
        images: Dict[str, np.ndarray],
        annotations: List[Dict[str, Any]],
        save_path: Optional[Path] = None
) -> List[Any]:

    assert isinstance(images, dict), "images must be a dict"
    assert isinstance(annotations, list), "annotations must be a list"

    if save_path is not None:
        save_path = Path(save_path)

    dataset = []
    for features in tqdm(annotations):
        image = images[features['filename']]
        h, w = image.shape[:2]

        boxes = []
        labels = []
        for b in features['boxes']:
            top, left = b['top'], b['left']
            height, width = b['height'], b['width']
            box = [top, left, top + height, left + width]
            labels.append(b['label'])
            boxes.append(box)

        labels = np.array(labels)
        weights = np.ones_like(labels)
        boxes = np.array(boxes) / np.array([h, w, h, w])

        features = Features(image=image)
        labels = LabelsFrame(
            boxes=boxes,
            labels=labels,
            weights=weights
        )
        dataset.append(ImageData(features, labels).to_dict())

    if save_path is not None:
        print(f"Saving dataset to: {save_path}")
        pickle.dump(dataset, save_path.open("wb"))

    return dataset


class DigitStructFile:
    """
    This class has been copied from the internet I don't remember the link to this
    implementation


    Example usage:

    dsf = DigitStructFile(fin)
    dataset = dsf.load()
    fout = open(options.filePrefix + ".json",'w')
    fout.write(JSONEncoder(indent = True).encode(dataset))
    fout.close()


    """
    def __init__(self, inf):
        """
        The DigitStructFile is just a wrapper around the h5py data.  It basically references
           inf:              The input h5 matlab file
           digitStructName   The h5 ref to all the file names
           digitStructBbox   The h5 ref to all struct data
        """
        self.inf = h5py.File(inf, "r")
        self.digit_struct_name = self.inf["digitStruct"]["name"]
        self.digit_struct_box = self.inf["digitStruct"]["bbox"]

    def get_name(self, n: int) -> str:
        """getName returns the 'name' string for for the n(th) digitStruct."""
        return "".join([chr(c[0]) for c in self.inf[self.digit_struct_name[n][0]].value])

    def bbox_helper(self, attr):
        """bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox."""
        if len(attr) > 1:
            attr = [
                self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))
            ]
        else:
            attr = [attr.value[0][0]]
        return attr

    def get_bbox(self, n: int) -> Dict[str, Any]:
        """# getBbox returns a dict of data for the n(th) bbox."""
        bbox = {}
        bb = self.digit_struct_box[n].item()
        bbox["height"] = self.bbox_helper(self.inf[bb]["height"])
        bbox["label"] = self.bbox_helper(self.inf[bb]["label"])
        bbox["left"] = self.bbox_helper(self.inf[bb]["left"])
        bbox["top"] = self.bbox_helper(self.inf[bb]["top"])
        bbox["width"] = self.bbox_helper(self.inf[bb]["width"])
        return bbox

    def get_digit_structure(self, n):
        s = self.get_bbox(n)
        s["name"] = self.get_name(n)
        return s

    def get_all_digit_structure(self):
        """
        # getAllDigitStructure returns all the digitStruct from the input file.

        """
        return [self.get_digit_structure(i) for i in range(len(self.digit_struct_name))]

    def load(self):
        """
        Return a restructured version of the dataset (one structure by boxed digit).

          Return a list of such dicts :
             'filename' : filename of the samples
             'boxes' : list of such dicts (one by digit) :
                 'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
                 'left', 'top' : position of bounding box
                 'width', 'height' : dimension of bounding box

        Note: We may turn this to a generator, if memory issues arise.
        """
        pictDat = self.get_all_digit_structure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = {"filename": pictDat[i]["name"]}
            figures = []
            for j in range(len(pictDat[i]["height"])):
                figure = {
                    "height": pictDat[i]["height"][j],
                    "label": pictDat[i]["label"][j],
                    "left": pictDat[i]["left"][j],
                    "top": pictDat[i]["top"][j],
                    "width": pictDat[i]["width"][j],
                }
                figures.append(figure)
            structCnt = structCnt + 1
            item["boxes"] = figures
            result.append(item)
        return result
